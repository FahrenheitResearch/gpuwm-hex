! MPAS: module_ra_rrtmg_lw.F:1712-2030. Test every public RNG entry point.
program randoms
use MersenneTwister
use mcica_random_numbers
use lw_binio
implicit none
integer,parameter::n=700
type(randomNumberSequence)::sequence
type(randomNumberStream)::stream
integer::scenario,k,scalar,iv(n),positive(n),state(625),seed4(0:3),seed700(0:699)
real::values(n),one,small(49),matrix(5,130)
character(20)::prefix
character(1024)::outdir
call get_command_argument(1,outdir)
seed4=[291,564,837,1110]
do k=0,699
 seed700(k)=k*7919-1000000
enddo
call bio_open(trim(outdir)//'/native-randoms.bin')
do scenario=0,5
 write(prefix,'(i3.3)')scenario
 select case(scenario)
 case(0);scalar=0
 case(1,4);scalar=5489
 case(2);scalar=-1
 case default;scalar=0
 end select
 if(scenario<4)then
  if(scenario==3)then
   sequence=new_RandomNumberSequence(seed4)
   call wr_i1(trim(prefix)//'/seed',seed4)
  else
   sequence=new_RandomNumberSequence(scalar)
   call wr_i1(trim(prefix)//'/seed',[scalar])
  endif
  do k=1,n
   iv(k)=getRandomInt(sequence)
  enddo
  do k=1,n
   positive(k)=getRandomPositiveInt(sequence)
  enddo
  do k=1,n
   values(k)=getRandomReal(sequence)
  enddo
  state=[sequence%currentElement,sequence%state]
  call wr_i1(trim(prefix)//'/integers',iv)
  call wr_i1(trim(prefix)//'/positive',positive)
  call wr_i1(trim(prefix)//'/state',state)
  call finalize_RandomNumberSequence(sequence)
  state=[sequence%currentElement,sequence%state]
  call wr_i1(trim(prefix)//'/finalized',state)
 else
  if(scenario==4)then
   stream=initializeRandomNumberStream(scalar)
   call wr_i1(trim(prefix)//'/seed',[scalar])
  else
   stream=initializeRandomNumberStream(seed700)
   call wr_i1(trim(prefix)//'/seed',seed700)
  endif
  call getRandomNumbers(stream,one)
  call getRandomNumbers(stream,small)
  call getRandomNumbers(stream,matrix)
  values=[one,small,reshape(matrix,[650])]
 endif
 call wr_r1(trim(prefix)//'/real',values)
enddo
call bio_close()
end program randoms
