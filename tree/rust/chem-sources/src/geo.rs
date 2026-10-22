// Deterministic spherical nearest-cell index.
use rayon::prelude::*;
pub const R: f64 = 6_371_229.;
pub fn xyz(lat: f64, lon: f64) -> [f64; 3] {
    let (s, c) = lat.sin_cos();
    let (sl, cl) = lon.sin_cos();
    [c * cl, c * sl, s]
}
struct Node {
    point: [f64; 3],
    id: usize,
    axis: usize,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
}
pub struct Index(Option<Box<Node>>);
impl Index {
    pub fn new(points: &[[f64; 3]]) -> Self {
        let mut rows: Vec<_> = points.iter().copied().enumerate().collect();
        Self(build(&mut rows, 0))
    }
    pub fn nearest(&self, p: [f64; 3]) -> (usize, f64) {
        let mut best = (usize::MAX, f64::INFINITY);
        search(&self.0, p, &mut best);
        (best.0, 2. * R * (best.1.sqrt() / 2.).min(1.).asin())
    }
    pub fn plan(&self, points: &[[f64; 3]]) -> Vec<(usize, f64)> {
        points.par_iter().map(|&p| self.nearest(p)).collect()
    }
}
fn build(rows: &mut [(usize, [f64; 3])], depth: usize) -> Option<Box<Node>> {
    if rows.is_empty() {
        return None;
    }
    let axis = depth % 3;
    let mid = rows.len() / 2;
    rows.select_nth_unstable_by(mid, |a, b| {
        a.1[axis].total_cmp(&b.1[axis]).then(a.0.cmp(&b.0))
    });
    let (left, rest) = rows.split_at_mut(mid);
    let (point, right) = rest.split_first_mut().unwrap();
    Some(Box::new(Node {
        point: point.1,
        id: point.0,
        axis,
        left: build(left, depth + 1),
        right: build(right, depth + 1),
    }))
}
fn search(node: &Option<Box<Node>>, p: [f64; 3], best: &mut (usize, f64)) {
    if let Some(n) = node {
        let d = (0..3).map(|i| (p[i] - n.point[i]).powi(2)).sum::<f64>();
        if d < best.1 || (d == best.1 && n.id < best.0) {
            *best = (n.id, d)
        }
        let delta = p[n.axis] - n.point[n.axis];
        let (first, other) = if delta <= 0. {
            (&n.left, &n.right)
        } else {
            (&n.right, &n.left)
        };
        search(first, p, best);
        if delta * delta <= best.1 {
            search(other, p, best)
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dateline_and_tie() {
        let i = Index::new(&[xyz(0., 179.9f64.to_radians()), xyz(0., 0.)]);
        assert_eq!(i.nearest(xyz(0., (-179.9f64).to_radians())).0, 0);
        let j = Index::new(&[[1., 0., 0.], [1., 0., 0.]]);
        assert_eq!(j.nearest([1., 0., 0.]).0, 0);
    }
}
