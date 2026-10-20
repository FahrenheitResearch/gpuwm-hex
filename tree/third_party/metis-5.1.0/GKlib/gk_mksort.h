/*
 * gk_mksort.h: the GK_MKQSORT in-place sort template GKlib and METIS use.
 *
 * Copyright 2026 the gpuwm-hex authors.
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy
 * of the License at http://www.apache.org/licenses/LICENSE-2.0. Unless
 * required by applicable law or agreed to in writing, software distributed
 * under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied.
 *
 * Written for gpuwm-hex on 2026-10-16. It replaces METIS 5.1.0's original
 * header, which was LGPL-2.1+ code from the GNU C Library. A public hex
 * package carries third-party code only under BSD, MIT, Apache-2.0 or public
 * domain (HEX-VENDORING.txt, change 3).
 *
 * Interface (unchanged): GK_MKQSORT(TYPE, BASE, NELT, LT) sorts the NELT
 * elements of type TYPE at BASE into the order LT(a, b) defines, where LT
 * takes two TYPE pointers and is true when *a must come before *b.
 *
 * Method: quicksort with a median-of-three pivot and an explicit stack that
 * always defers the larger part (R. Sedgewick, "Implementing Quicksort
 * Programs", CACM 21(10), 1978). Parts of at most GK_QS_CUTOFF + 1 elements
 * are not partitioned further. A single insertion pass over the whole array
 * then finishes it, starting from a sentinel minimum taken from the first
 * GK_QS_CUTOFF + 1 elements. The sort is not stable. It places equal keys
 * exactly where METIS 5.1.0's own build places them, so the partitions METIS
 * writes are unchanged (HEX-VENDORING.txt, change 3, names the measurement).
 */

#ifndef _GK_MKSORT_H_
#define _GK_MKSORT_H_

#include <stddef.h>

/* Parts with at most this many elements past their first are left to the
 * final insertion pass. */
#define GK_QS_CUTOFF 4
/* The smaller part is always the one deferred, so the stack never holds more
 * than log2(NELT) entries. */
#define GK_QS_DEPTH (8 * sizeof(size_t))

#define GK_QS_EXCHANGE(T, x, y)                                             \
  do { T gkq_swap_ = *(x); *(x) = *(y); *(y) = gkq_swap_; } while (0)

#define GK_MKQSORT(GKQ_TYPE, GKQ_BASE, GKQ_NELT, GKQ_LT)                   \
do {                                                                        \
  GKQ_TYPE *const gkq_first = (GKQ_BASE);                                   \
  const size_t gkq_count = (size_t)(GKQ_NELT);                              \
                                                                            \
  /* 1. Partition until every part is small. */                            \
  if (gkq_count > GK_QS_CUTOFF) {                                           \
    GKQ_TYPE *gkq_lo = gkq_first;                                           \
    GKQ_TYPE *gkq_hi = gkq_first + gkq_count - 1;                           \
    GKQ_TYPE *gkq_pend_lo[GK_QS_DEPTH];                                     \
    GKQ_TYPE *gkq_pend_hi[GK_QS_DEPTH];                                     \
    size_t gkq_pending = 0;                                                 \
                                                                            \
    for (;;) {                                                              \
      GKQ_TYPE *gkq_pivot = gkq_lo + ((gkq_hi - gkq_lo) >> 1);              \
      GKQ_TYPE *gkq_up;                                                     \
      GKQ_TYPE *gkq_down;                                                   \
      int gkq_small_low, gkq_small_high;                                    \
                                                                            \
      /* Order lo <= pivot <= hi; the pivot is their median. */             \
      if (GKQ_LT(gkq_pivot, gkq_lo))                                        \
        GK_QS_EXCHANGE(GKQ_TYPE, gkq_pivot, gkq_lo);                        \
      if (GKQ_LT(gkq_hi, gkq_pivot)) {                                      \
        GK_QS_EXCHANGE(GKQ_TYPE, gkq_pivot, gkq_hi);                        \
        if (GKQ_LT(gkq_pivot, gkq_lo))                                      \
          GK_QS_EXCHANGE(GKQ_TYPE, gkq_pivot, gkq_lo);                      \
      }                                                                     \
                                                                            \
      /* Scan inward from both ends, exchanging out-of-place pairs. The    \
       * pivot element can itself move; follow it. */                       \
      gkq_up = gkq_lo + 1;                                                  \
      gkq_down = gkq_hi - 1;                                                \
      do {                                                                  \
        while (GKQ_LT(gkq_up, gkq_pivot))                                   \
          ++gkq_up;                                                         \
        while (GKQ_LT(gkq_pivot, gkq_down))                                 \
          --gkq_down;                                                       \
        if (gkq_up < gkq_down) {                                            \
          GK_QS_EXCHANGE(GKQ_TYPE, gkq_up, gkq_down);                       \
          if (gkq_pivot == gkq_up)                                          \
            gkq_pivot = gkq_down;                                           \
          else if (gkq_pivot == gkq_down)                                   \
            gkq_pivot = gkq_up;                                             \
          ++gkq_up;                                                         \
          --gkq_down;                                                       \
        } else {                                                            \
          if (gkq_up == gkq_down) {                                         \
            ++gkq_up;                                                       \
            --gkq_down;                                                     \
          }                                                                 \
          break;                                                            \
        }                                                                   \
      } while (gkq_up <= gkq_down);                                         \
                                                                            \
      /* Two parts remain: [lo, down] and [up, hi]. */                      \
      gkq_small_low = (size_t)(gkq_down - gkq_lo) <= GK_QS_CUTOFF;          \
      gkq_small_high = (size_t)(gkq_hi - gkq_up) <= GK_QS_CUTOFF;           \
      if (gkq_small_low && gkq_small_high) {                                \
        if (gkq_pending == 0)                                               \
          break;                                                            \
        --gkq_pending;                                                      \
        gkq_lo = gkq_pend_lo[gkq_pending];                                  \
        gkq_hi = gkq_pend_hi[gkq_pending];                                  \
      } else if (gkq_small_low) {                                           \
        gkq_lo = gkq_up;                                                    \
      } else if (gkq_small_high) {                                          \
        gkq_hi = gkq_down;                                                  \
      } else if ((gkq_down - gkq_lo) > (gkq_hi - gkq_up)) {                 \
        gkq_pend_lo[gkq_pending] = gkq_lo;                                  \
        gkq_pend_hi[gkq_pending] = gkq_down;                                \
        ++gkq_pending;                                                      \
        gkq_lo = gkq_up;                                                    \
      } else {                                                              \
        gkq_pend_lo[gkq_pending] = gkq_up;                                  \
        gkq_pend_hi[gkq_pending] = gkq_hi;                                  \
        ++gkq_pending;                                                      \
        gkq_hi = gkq_down;                                                  \
      }                                                                     \
    }                                                                       \
  }                                                                         \
                                                                            \
  /* 2. One insertion pass. The array minimum is among the first           \
   * GK_QS_CUTOFF + 1 elements, so it goes to the front as a sentinel and   \
   * the inner scan needs no bounds check. */                               \
  if (gkq_count > 1) {                                                      \
    GKQ_TYPE *const gkq_last = gkq_first + gkq_count - 1;                   \
    GKQ_TYPE *gkq_least = gkq_first;                                        \
    GKQ_TYPE *gkq_scan_end = gkq_first + GK_QS_CUTOFF;                      \
    GKQ_TYPE *gkq_next;                                                     \
    if (gkq_scan_end > gkq_last)                                            \
      gkq_scan_end = gkq_last;                                              \
    for (gkq_next = gkq_first + 1; gkq_next <= gkq_scan_end; ++gkq_next)   \
      if (GKQ_LT(gkq_next, gkq_least))                                      \
        gkq_least = gkq_next;                                               \
    if (gkq_least != gkq_first)                                             \
      GK_QS_EXCHANGE(GKQ_TYPE, gkq_least, gkq_first);                       \
                                                                            \
    for (gkq_next = gkq_first + 2; gkq_next <= gkq_last; ++gkq_next) {     \
      GKQ_TYPE *gkq_slot = gkq_next - 1;                                    \
      while (GKQ_LT(gkq_next, gkq_slot))                                    \
        --gkq_slot;                                                         \
      ++gkq_slot;                                                           \
      if (gkq_slot != gkq_next) {                                           \
        GKQ_TYPE gkq_moving = *gkq_next;                                    \
        GKQ_TYPE *gkq_move;                                                 \
        for (gkq_move = gkq_next; gkq_move > gkq_slot; --gkq_move)          \
          *gkq_move = *(gkq_move - 1);                                      \
        *gkq_slot = gkq_moving;                                             \
      }                                                                     \
    }                                                                       \
  }                                                                         \
} while (0)

#endif /* _GK_MKSORT_H_ */
