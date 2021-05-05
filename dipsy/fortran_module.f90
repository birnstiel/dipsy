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
        doubleprecision, intent(IN) :: pars(7)
        doubleprecision, intent(IN) :: x(nx)
        doubleprecision, intent(OUT):: y(nx)
        doubleprecision :: r0
        doubleprecision :: n = 0.1

        ! this function has 7 parameters:
        ! 1: normalization
        ! 2: brightness drop at dust line
        ! 3: outer power-law r**-p
        ! 4: inner power-law r**-p
        ! 5: position of outer exponential taper
        ! 6: position of power-law transition
        ! 7: position of dust line

        ! logistic taper by factor pars(2) at position r0 with
        ! a width of 0.1 * r0
        ! r0 is the center of the logit function, but we want
        ! the parameter pars(7) to be the "outer end" of that logit
        ! transition, so we define this at being 90% in log space between
        ! the left (=1) and the right (=1-pars(2)) limit of the logit function.

        if (pars(2) > 1e-8) then
            r0 = pars(7)/(1d0 + 0.1d0*log(pars(2)/((1d0 - pars(2))**n + pars(2) - 1d0) - 1d0))
        else
            r0 = pars(7)/(1d0 + 0.1d0*log(n/(1d0 - n)))
        end if

        y = 1 - pars(2)/(1d0 + exp(-(x - r0)/(0.1*r0)))

        ! two power laws smoothly connected at pars(6)

        y = y*pars(1)*(((x/pars(6))**-pars(4))**-2 + ((x/pars(6))**-pars(3))**-2)**(-0.5)

        ! exponential taper at pars(5)

        y = y*exp(-(x/pars(5))**4)
    end subroutine pwr2_logit

    subroutine lbp_profile(pars, x, y, nx)
        implicit none
        integer, intent(IN) :: nx
        doubleprecision, intent(IN) :: pars(3)
        doubleprecision, intent(IN) :: x(nx)
        doubleprecision, intent(OUT):: y(nx)
        ! this is the lynden-bell & pringle self-similar profile
        ! the three parameters are
        ! 1: normalization
        ! 2: characteristic radius
        ! 3: viscosity index, i.e. p in nu \propto r**p

        y = pars(1)*(x/pars(2))**-pars(3)*exp(-(x/pars(2))**(2 - pars(3)))

    end subroutine lbp_profile

    doubleprecision function lnp_pwr(params, x, y, nx)
        implicit none
        integer, intent(IN) :: nx
        doubleprecision, intent(IN) :: params(3)
        doubleprecision, intent(IN) :: x(nx), y(nx)
        doubleprecision :: ym(nx)

        if ( &
            (params(1) < 0d0) &
            .or. (params(2) < -10) &
            .or. (params(2) > 10) &
            .or. (params(3) < x(1)) &
            .or. (params(3) > x(nx))) then
            lnp_pwr = inf_neg()
        else
            call pwr1(params, x, ym, nx)
            lnp_pwr = sum(-(MIN(1d100, ym - y)**2/(2d0*(MAX(crop, 0.1d0*y))**2)))
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

            lnp_pwr2 = sum(-(MIN(1d100, ym - y)**2/(2d0*(MAX(crop, 0.1d0*y))**2)))
            if (lnp_pwr2 .ne. lnp_pwr2) then
                lnp_pwr2 = inf_neg()
            end if
        end if

    end function lnp_pwr2

    doubleprecision function lnp_pwr2_logit(params, x, y, nx)
        implicit none
        integer, intent(IN) :: nx
        doubleprecision, intent(IN) :: params(7)
        doubleprecision, intent(IN) :: x(nx), y(nx)
        doubleprecision :: ym(nx)
        ! this function has 6 parameters:
        ! 1: normalization
        ! 2: brightness drop at dust line
        ! 3: outer power-law r**-p
        ! 4: inner power-law r**-p
        ! 5: position of exponential taper
        ! 6: position of power-law transition
        ! 7: position of dust line

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
            .or. (params(7) < x(1)) &
            .or. (params(7) > x(nx)) &
            .or. (params(6) > params(5)) &
            .or. (params(7) > params(5)) &
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

    doubleprecision function lnp_lbp(params, x, y, nx)
        implicit none
        integer, intent(IN) :: nx
        doubleprecision, intent(IN) :: params(3)
        doubleprecision, intent(IN) :: x(nx), y(nx)
        doubleprecision :: ym(nx)
        ! this function has 3 parameters:
        ! 1: normalization
        ! 2: characteristic radius
        ! 3: viscosity exponent

        if ( &
            (params(1) < 0) &
            .or. (params(2) <= 0) &
            .or. (params(3) < -10) &
            .or. (params(3) > 10) &
            ) then
            lnp_lbp = inf_neg()
        else
            call lbp_profile(params, x, ym, nx)

            lnp_lbp = sum(-(MIN(1d100, ym - y)**2/(2d0*(MAX(crop, 0.1d0*y))**2)))
            if (lnp_lbp .ne. lnp_lbp) then
                lnp_lbp = inf_neg()
            end if
        end if

    end function lnp_lbp

end module fmodule
