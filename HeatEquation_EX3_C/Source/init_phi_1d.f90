subroutine init_phi(lo, hi, phi, philo, phihi, dx, prob_lo, prob_hi) bind(C, name="init_phi")

  use amrex_fort_module, only : amrex_real

  implicit none

  integer, intent(in) :: lo(1), hi(1), philo(1), phihi(1)
  real(amrex_real), intent(inout) :: phi(philo(1):phihi(1))
  real(amrex_real), intent(in   ) :: dx(1)
  real(amrex_real), intent(in   ) :: prob_lo(1)
  real(amrex_real), intent(in   ) :: prob_hi(1)

    integer          :: i
  double precision :: x, sqrt_0_13

  sqrt_0_13 = sqrt(0.13d0) 

     do i = lo(1), hi(1)
        x = prob_lo(1) + (dble(i)+0.5d0) * dx(1)
         
         if (x >= sqrt_0_13 .or. x <= -sqrt_0_13) then  
            phi(i) = 0.0d0  
         else
            phi(i) = sqrt((1.0d0 / 6.0d0)*(1.3d0 - x**2 / 0.1d0)) /  0.1d0 
         end if  
     end do
 

end subroutine init_phi
