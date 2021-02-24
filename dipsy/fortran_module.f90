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

        if ( &
            (params(1) < 0) &
            .or. (params(2) < -10) &
            .or. (params(2) > 10) &
            .or. (params(3) < x(1)) &
            .or. (params(3) > x(nx))) then
            lnp_pwr = inf_neg()
        else
            call pwr1(params, x, ym, nx)
            lnp_pwr = sum(-(MIN(1d300, ym - y)**2/(1d-100 + 2d0*(0.1d0*y)**2)))
            if (lnp_pwr .ne. lnp_pwr) then
                lnp_pwr = inf_neg()
            end if
        end if

    end function lnp_pwr

    subroutine pwr1(pars, x, y, nx)
        implicit none
        integer, intent(IN) :: nx
        doubleprecision, intent(IN) :: pars(3)
        doubleprecision, intent(IN) :: x(nx)
        doubleprecision, intent(OUT):: y(nx)

        where (x < pars(3))
            y = pars(1)*x**pars(2)
        elsewhere
            y = 0
        end where
    end subroutine pwr1

    subroutine pwr2(pars, x, y, nx)
        implicit none
        integer, intent(IN) :: nx
        doubleprecision, intent(IN) :: pars(6)
        doubleprecision, intent(IN) :: x(nx)
        doubleprecision, intent(OUT):: y(nx)

        where (x < pars(6))
            where (x < pars(3))
                y = pars(1)*(x/pars(3))**(-pars(2))
            elsewhere
                y = pars(1)*pars(4)*(x/pars(3))**(-pars(5))
            end where
        elsewhere
            y = 0
        end where
    end subroutine pwr2

    doubleprecision function lnp_pwr2(params, x, y, nx)
        implicit none
        integer, intent(IN) :: nx
        doubleprecision, intent(IN) :: params(6)
        doubleprecision, intent(IN) :: x(nx), y(nx)
        doubleprecision :: ym(nx)
        ! this function has 6 parameters:
        ! 1: Normalization N
        ! 2: power-law 1
        ! 3: brightness drop position
        ! 4: brightness drop depth
        ! 5: power-law 2
        ! 6: outer edge

        if ( &
            (params(1) < 0) &
            .or. (params(2) < -10) &
            .or. (params(2) > 10) &
            .or. (params(3) < x(1)) &
            .or. (params(3) > x(nx)) &
            .or. (params(4) < 0) &
            .or. (params(4) > 1) &
            .or. (params(5) < -10) &
            .or. (params(5) > 10) &
            .or. (params(6) < x(1)) &
            .or. (params(6) > x(nx))) then
            lnp_pwr2 = inf_neg()
        else
            call pwr2(params, x, ym, nx)

            lnp_pwr2 = sum(-(MIN(1d300, ym - y)**2/(1d-100 + 2d0*(0.1d0*y)**2)))
            if (lnp_pwr2 .ne. lnp_pwr2) then
                lnp_pwr2 = inf_neg()
            end if
        end if

    end function lnp_pwr2

end module fmodule
