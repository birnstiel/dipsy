module fmodule
    implicit none
    doubleprecision :: crop = 1e-10
contains

    function inf_neg() result(r)
        use, intrinsic :: ieee_arithmetic
        real(8) :: r
        r = ieee_value(0d0, ieee_negative_inf)
    end function inf_neg

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
        ! this function has 6 parameters:
        ! 1: Normalization N
        ! 2: power-law 1
        ! 3: power-law 2
        ! 4: outer edge
        ! 5: brightness drop depth
        ! 6: brightness drop position
        where (x < pars(4))
            where (x < pars(6))
                y = pars(1)*(x/pars(6))**(-pars(2))
            elsewhere
                y = pars(1)*pars(5)*(x/pars(6))**(-pars(3))
            end where
        elsewhere
            y = 0
        end where
    end subroutine pwr2

    subroutine pwr2_logit(pars, x, y, nx)
        implicit none
        integer, intent(IN) :: nx
        doubleprecision, intent(IN) :: pars(6)
        doubleprecision, intent(IN) :: x(nx)
        doubleprecision, intent(OUT):: y(nx)
        ! this function has 6 parameters:
        ! 1: normalization
        ! 2: brightness drop at dust line
        ! 3: outer power-law r**-p
        ! 4: inner power-law r**-p
        ! 5: outer exponential taper
        ! 6: dust line position

        ! logistic taper by factor pars(2) at position pars(6) with
        ! a width of 0.1 * pars(6)

        y = 1 - pars(2)/(1d0 + exp(-(x - pars(6))/(0.1*pars(6))))

        ! two power laws smoothly connected

        y = y*pars(1)*(((x/pars(6))**-pars(4))**-2 + ((x/pars(6))**-pars(3))**-2)**(-0.5)

        ! exponential taper

        y = y*exp(-(x/pars(5))**4)
    end subroutine pwr2_logit

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
            lnp_pwr = sum(-(MIN(1d100, ym - y)**2/(2d0*(MIN(crop, 0.1d0*y))**2)))
            if (lnp_pwr .ne. lnp_pwr) then
                lnp_pwr = inf_neg()
            end if
        end if

    end function lnp_pwr

    doubleprecision function lnp_pwr2(params, x, y, nx)
        implicit none
        integer, intent(IN) :: nx
        doubleprecision, intent(IN) :: params(6)
        doubleprecision, intent(IN) :: x(nx), y(nx)
        doubleprecision :: ym(nx)

        if ( &
            (params(1) < 0) &
            .or. (params(2) < -10) &
            .or. (params(2) > 10) &
            .or. (params(3) < -10) &
            .or. (params(3) > 10) &
            .or. (params(4) < x(1)) &
            .or. (params(4) > x(nx)) &
            .or. (params(5) < 0) &
            .or. (params(5) > 1) &
            .or. (params(6) < x(1)) &
            .or. (params(6) > x(nx)) &
            .or. (params(6) > params(4)) &
            ) then
            lnp_pwr2 = inf_neg()
        else
            call pwr2(params, x, ym, nx)

            lnp_pwr2 = sum(-(MIN(1d100, ym - y)**2/(2d0*(MIN(crop, 0.1d0*y))**2)))
            if (lnp_pwr2 .ne. lnp_pwr2) then
                lnp_pwr2 = inf_neg()
            end if
        end if

    end function lnp_pwr2

    doubleprecision function lnp_pwr2_logit(params, x, y, nx)
        implicit none
        integer, intent(IN) :: nx
        doubleprecision, intent(IN) :: params(6)
        doubleprecision, intent(IN) :: x(nx), y(nx)
        doubleprecision :: ym(nx)
        ! this function has 6 parameters:
        ! 1: normalization
        ! 2: brightness drop at dust line
        ! 3: outer power-law r**-p
        ! 4: inner power-law r**-p
        ! 5: outer exponential taper
        ! 6: dust line position

        if ( &
            (params(1) < 0) &
            .or. (params(2) < 0) &
            .or. (params(2) > 1) &
            .or. (params(3) < -10) &
            .or. (params(3) > 10) &
            .or. (params(4) < -10) &
            .or. (params(4) > 10) &
            .or. (params(5) < x(1)) &
            .or. (params(5) > x(nx)) &
            .or. (params(6) < x(1)) &
            .or. (params(6) > x(nx)) &
            .or. (params(6) > params(5)) &
            ) then
            lnp_pwr2_logit = inf_neg()
        else
            call pwr2_logit(params, x, ym, nx)

            lnp_pwr2_logit = sum(-(MIN(1d100, ym - y)**2/(2d0*(MAX(crop, 0.1d0*y))**2)))
            if (lnp_pwr2_logit .ne. lnp_pwr2_logit) then
                lnp_pwr2_logit = inf_neg()
            end if
        end if

    end function lnp_pwr2_logit

end module fmodule
