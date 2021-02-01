module fmodule
    implicit none
contains

    function inf_neg() result(r)
        use, intrinsic :: ieee_arithmetic
        real(8) :: r
        r = ieee_value(0d0, ieee_negative_inf)
    end function inf_neg

    doubleprecision function lnp_pwr(params, x, y, nx)
        implicit none
        integer, intent(IN) :: nx
        doubleprecision, intent(IN) :: params(3)
        doubleprecision, intent(IN) :: x(nx), y(nx)
        doubleprecision :: ym(nx)

        if ((params(1) < -10) .or. (params(1) > 10)) then
            lnp_pwr = inf_neg()
        elseif ((params(3) < x(1)) .or. (params(3) > x(nx))) then
            lnp_pwr = inf_neg()
        else
            where (x < params(3))
                ym = params(1)*x**params(2)
            elsewhere
                ym = 0
            end where

            lnp_pwr = sum(-(MIN(1d300, ym - y)**2/(1d-100 + 2d0*(0.1d0*y)**2)))
            if (lnp_pwr .ne. lnp_pwr) then
                lnp_pwr = inf_neg()
            end if
        end if

    end function lnp_pwr

end module fmodule
