using LinearAlgebra

function init_slacks!(solver::MadNLP.MadNLPSolver)
    dlb = MadNLP.dual_lb(solver.d)
    @inbounds @simd for i in eachindex(dlb)
        solver.zl_r[i] = max(1.0, abs(solver.zl_r[i] + dlb[i]))
    end

    dub = MadNLP.dual_ub(solver.d)
    @inbounds @simd for i in eachindex(dub)
        solver.zu_r[i] = max(1.0, abs(solver.zu_r[i] + dub[i]))
    end
    return
end

function l_initialize!(solver::MadNLP.AbstractMadNLPSolver{T}) where T
    # initializing slack variables
    NLPModels.cons!(solver.nlp,MadNLP.get_x0(solver.nlp),MadNLP._madnlp_unsafe_wrap(solver.c,MadNLP.get_ncon(solver.nlp)))
    solver.cnt.con_cnt += 1
    copyto!(MadNLP.slack(solver.x), solver.c_slk)

    # Initialization
    fill!(solver.zl_r, one(T))
    fill!(solver.zu_r, one(T))

    MadNLP.set_initial_bounds!(solver)
    copyto!(MadNLP.primal(solver.x), MadNLP.get_x0(solver.nlp))

    # Automatic scaling (constraints)
    MadNLP.eval_jac_wrapper!(solver, solver.kkt, solver.x)
    MadNLP.compress_jacobian!(solver.kkt)

    MadNLP.eval_grad_f_wrapper!(solver, solver.f,solver.x)

    # Initializing
    solver.obj_val = MadNLP.eval_f_wrapper(solver, solver.x)
    MadNLP.eval_cons_wrapper!(solver, solver.c, solver.x)
    MadNLP.eval_lag_hess_wrapper!(solver, solver.kkt, solver.x, solver.y)

    # Change initial variable
    # MadNLP.jtprod!(solver.jacl, solver.kkt, solver.y)
    # MadNLP.set_aug_diagonal!(solver.kkt, solver)
    # MadNLP.factorize_wrapper!(solver)

    # set_predictive_rhs!(solver, solver.kkt)
    # MadNLP.solve_refine_wrapper!(solver, solver.d, solver.p)
    # MadNLP.finish_aug_solve!(solver, solver.kkt, 0.0)

    # init_slacks!(solver)


    theta = MadNLP.get_theta(solver.c)
    solver.theta_max = 1e4*max(1,theta)
    solver.theta_min = 1e-4*max(1,theta)
    solver.mu = solver.opt.mu_init
    solver.tau = max(solver.opt.tau_min,1-solver.opt.mu_init)
    solver.filter = [(solver.theta_max,-Inf)]

    return MadNLP.REGULAR
end

function get_complementarity_measure(solver::MadNLP.MadNLPSolver)
    m1, m2 = length(solver.x_lr), length(solver.x_ur)
    inf_compl = 0.0
    @inbounds @simd for i in 1:m1
        inf_compl += (solver.x_lr[i]-solver.xl_r[i])*solver.zl_r[i]
    end
    @inbounds @simd for i in 1:m2
        inf_compl += (solver.xu_r[i]-solver.x_ur[i])*solver.zu_r[i]
    end
    return inf_compl/ (m1 + m2)
end

function get_affine_complementarity_measure2(solver::MadNLP.MadNLPSolver, alpha::Float64)
    m1, m2 = length(solver.x_lr), length(solver.x_ur)
    dz1 =  MadNLP.dual_lb(solver.d)
    dz2 =  MadNLP.dual_ub(solver.d)

    inf_compl = 0.0
    @inbounds @simd for i in 1:m1
        inf_compl += (solver.x_trial_lr[i]-solver.xl_r[i])*(solver.zl_r[i] + alpha * dz1[i])
    end
    @inbounds @simd for i in 1:m2
        inf_compl += (solver.xu_r[i]-solver.x_trial_ur[i])*(solver.zu_r[i] + alpha * dz2[i])
    end

    return inf_compl/ (m1 + m2)
end

function get_affine_complementarity_measure(solver::MadNLP.MadNLPSolver, alpha_p::Float64, alpha_d::Float64)
    m1, m2 = length(solver.x_lr), length(solver.x_ur)
    dz1 =  MadNLP.dual_lb(solver.d)
    dz2 =  MadNLP.dual_ub(solver.d)

    inf_compl = 0.0
    @inbounds @simd for i in 1:m1
        x_lb = solver.xl_r[i]
        x_ = solver.x_lr[i] + alpha_p * solver.dx_lr[i]
        z_ = solver.zl_r[i] + alpha_d * dz1[i]
        inf_compl += (x_ - x_lb) * z_
    end
    @inbounds @simd for i in 1:m2
        x_ub = solver.xu_r[i]
        x_ = solver.x_ur[i] + alpha_p * solver.dx_ur[i]
        z_ = solver.zu_r[i] + alpha_d * dz2[i]
        inf_compl += (x_ub - x_) * z_
    end

    return inf_compl/ (m1 + m2)
end

function get_alpha_max_primal(x, xl, xu, dx, tau)
    alpha_max = 1.0
    @inbounds @simd for i=1:length(x)
        dx[i]<0 && (alpha_max=min(alpha_max,(-x[i]+xl[i])*tau/dx[i]))
        dx[i]>0 && (alpha_max=min(alpha_max,(-x[i]+xu[i])*tau/dx[i]))
    end
    return alpha_max
end

function get_alpha_max_dual(zl_r, zu_r, dzl, dzu, tau)
    alpha_z = 1.0
    @inbounds @simd for i=1:length(zl_r)
        dzl[i] < 0 && (alpha_z=min(alpha_z,-zl_r[i]*tau/dzl[i]))
     end
    @inbounds @simd for i=1:length(zu_r)
        dzu[i] < 0 && (alpha_z=min(alpha_z,-zu_r[i]*tau/dzu[i]))
    end
    return alpha_z
end

function set_predictive_rhs!(solver::MadNLP.MadNLPSolver, kkt::MadNLP.AbstractKKTSystem)
    # RHS
    px = MadNLP.primal(solver.p)
    py = MadNLP.dual(solver.p)
    # Gradient
    f = MadNLP.primal(solver.f)
    # Constraint
    c = solver.c
    @inbounds @simd for i in eachindex(px)
        px[i] = -f[i] - solver.jacl[i]
    end
    @inbounds @simd for i in eachindex(py)
        py[i] = -c[i]
    end
    return
end

function set_corrective_rhs!(solver::MadNLP.MadNLPSolver, kkt::MadNLP.AbstractKKTSystem, mu::Float64, correction_lb::Vector{Float64}, correction_ub::Vector{Float64}, ind_lb, ind_ub)
    px = MadNLP.primal(solver.p)
    x = MadNLP.primal(solver.x)
    f = MadNLP.primal(solver.f)
    xl = MadNLP.primal(solver.xl)
    xu = MadNLP.primal(solver.xu)
    @inbounds @simd for i in eachindex(px)
        px[i] = -f[i] - solver.jacl[i] + mu / (x[i] - xl[i]) - mu / (xu[i] - x[i])
    end
    py = MadNLP.dual(solver.p)
    @inbounds @simd for i in eachindex(py)
        py[i] = -solver.c[i]
    end

    for (k, i) in enumerate(ind_lb)
        px[i] -= correction_lb[k] / (x[i] - xl[i])
    end
    for (k, i) in enumerate(ind_ub)
        px[i] -= correction_ub[k] / (xu[i] - x[i])
    end

    return
end

function get_correction!(
    solver::MadNLP.MadNLPSolver,
    correction_lb,
    correction_ub,
    ind_lb,
    ind_ub,
)
    dx = MadNLP.primal(solver.d)
    dlb = MadNLP.dual_lb(solver.d)
    dub = MadNLP.dual_ub(solver.d)

    tol = 1e-8
    for i in eachindex(dlb)
        dd = solver.dx_lr[i] * dlb[i]
        if abs(dd) > tol
            correction_lb[i] = dd
        else
            correction_lb[i] = 0.0
        end
    end
    for i in eachindex(dub)
        dd = solver.dx_ur[i] * dub[i]
        if abs(dd) > tol
            correction_ub[i] = dd
        else
            correction_ub[i] = 0.0
        end
    end
    return
end

function finish_corrective_aug_solve!(solver::MadNLP.MadNLPSolver, kkt::MadNLP.AbstractKKTSystem, correction_lb, correction_ub, mu)
    dlb = MadNLP.dual_lb(solver.d)
    @assert length(dlb) == length(correction_lb)
    @inbounds @simd for i in eachindex(dlb)
        dlb[i] = (mu-correction_lb[i]-solver.zl_r[i]*solver.dx_lr[i])/(solver.x_lr[i]-solver.xl_r[i])-solver.zl_r[i]
    end
    dub = MadNLP.dual_ub(solver.d)
    @assert length(dub) == length(correction_ub)
    @inbounds @simd for i in eachindex(dub)
        dub[i] = (mu+correction_ub[i]+solver.zu_r[i]*solver.dx_ur[i])/(solver.xu_r[i]-solver.x_ur[i])-solver.zu_r[i]
    end
    return
end

# Predictor-corrector method
function mpc!(solver::MadNLP.AbstractMadNLPSolver)
    ind_cons = MadNLP.get_index_constraints(solver.nlp)
    nlb, nub = length(ind_cons.ind_lb), length(ind_cons.ind_ub)
    correction_lb = zeros(nlb)
    correction_ub = zeros(nub)

    while true
        # A' y
        MadNLP.jtprod!(solver.jacl, solver.kkt, solver.y)

        #####
        # Update info
        #####
        sd = MadNLP.get_sd(solver.y,solver.zl_r,solver.zu_r,solver.opt.s_max)
        sc = MadNLP.get_sc(solver.zl_r,solver.zu_r,solver.opt.s_max)
        solver.inf_pr = MadNLP.get_inf_pr(solver.c)
        solver.inf_du = MadNLP.get_inf_du(
            MadNLP.full(solver.f),
            MadNLP.full(solver.zl),
            MadNLP.full(solver.zu),
            solver.jacl,
            1.0,
        )
        solver.inf_compl = MadNLP.get_inf_compl(solver.x_lr,solver.xl_r,solver.zl_r,solver.xu_r,solver.x_ur,solver.zu_r,0.,sc)
        MadNLP.print_iter(solver)

        #####
        # Termination criteria
        #####
        if max(solver.inf_pr,solver.inf_du,solver.inf_compl) <= solver.opt.tol
            return MadNLP.SOLVE_SUCCEEDED
        elseif solver.cnt.k >= solver.opt.max_iter
            return MadNLP.MAXIMUM_ITERATIONS_EXCEEDED
        end


        #####
        # Factorize
        #####
        MadNLP.set_aug_diagonal!(solver.kkt, solver)
        MadNLP.factorize_wrapper!(solver)

        #####
        # Prediction step
        #####

        set_predictive_rhs!(solver, solver.kkt)
        MadNLP.solve_refine_wrapper!(solver, solver.d, solver.p)
        MadNLP.finish_aug_solve!(solver, solver.kkt, 0.0) # mu set to 0

        # Stepsize
        alpha_p = get_alpha_max_primal(
            MadNLP.primal(solver.x),
            MadNLP.primal(solver.xl),
            MadNLP.primal(solver.xu),
            MadNLP.primal(solver.d),
            1.0,
        )
        alpha_d = get_alpha_max_dual(
            solver.zl_r,
            solver.zu_r,
            MadNLP.dual_lb(solver.d),
            MadNLP.dual_ub(solver.d),
            1.0,
        )
        alpha_aff = min(alpha_p, alpha_d)

        # Primal step
        mu_affine = get_affine_complementarity_measure(solver, alpha_aff, alpha_aff)
        get_correction!(solver, correction_lb, correction_ub, ind_cons.ind_lb, ind_cons.ind_ub)

        #####
        # Update barrier
        #####
        # μ = y' s / m
        mu_curr = get_complementarity_measure(solver)
        sigma = (mu_affine / mu_curr)^3
        sigma = clamp(sigma, 1e-6, 10.0 )
        mu = max(solver.opt.mu_min, sigma * mu_curr)
        tau = MadNLP.get_tau(mu, solver.opt.tau_min)
        solver.mu = mu

        #####
        # Correction step
        #####
        set_corrective_rhs!(solver, solver.kkt, mu, correction_lb, correction_ub, ind_cons.ind_lb, ind_cons.ind_ub)
        MadNLP.solve_refine_wrapper!(solver, solver.d, solver.p)
        finish_corrective_aug_solve!(solver, solver.kkt, correction_lb, correction_ub, mu)

        alpha_p = get_alpha_max_primal(
            MadNLP.primal(solver.x),
            MadNLP.primal(solver.xl),
            MadNLP.primal(solver.xu),
            MadNLP.primal(solver.d),
            tau,
        )
        alpha_d = get_alpha_max_dual(
            solver.zl_r,
            solver.zu_r,
            MadNLP.dual_lb(solver.d),
            MadNLP.dual_ub(solver.d),
            tau,
        )
        solver.alpha = 0.9995 * min(alpha_p, alpha_d)
        solver.alpha_z = solver.alpha

        #####
        # Next trial point
        #####
        # Update primal-dual solution
        axpy!(solver.alpha, MadNLP.primal(solver.d), MadNLP.primal(solver.x))
        axpy!(solver.alpha, MadNLP.dual(solver.d), solver.y)
        solver.zl_r .+= solver.alpha_z .* MadNLP.dual_lb(solver.d)
        solver.zu_r .+= solver.alpha_z .* MadNLP.dual_ub(solver.d)

        solver.obj_val = MadNLP.eval_f_wrapper(solver, solver.x)
        MadNLP.eval_cons_wrapper!(solver, solver.c, solver.x)
        MadNLP.eval_grad_f_wrapper!(solver, solver.f, solver.x)


        solver.cnt.k+=1
    end
end

function l_regular!(solver::MadNLP.AbstractMadNLPSolver{T}) where T
    while true
        MadNLP.jtprod!(solver.jacl, solver.kkt, solver.y)

        sd = MadNLP.get_sd(solver.y,solver.zl_r,solver.zu_r,solver.opt.s_max)
        sc = MadNLP.get_sc(solver.zl_r,solver.zu_r,solver.opt.s_max)

        solver.inf_pr = MadNLP.get_inf_pr(solver.c)
        solver.inf_du = MadNLP.get_inf_du(
            MadNLP.full(solver.f),
            MadNLP.full(solver.zl),
            MadNLP.full(solver.zu),
            solver.jacl,
            sd,
        )
        solver.inf_compl = MadNLP.get_inf_compl(solver.x_lr,solver.xl_r,solver.zl_r,solver.xu_r,solver.x_ur,solver.zu_r,0.,sc)
        inf_compl_mu = MadNLP.get_inf_compl(solver.x_lr,solver.xl_r,solver.zl_r,solver.xu_r,solver.x_ur,solver.zu_r,solver.mu,sc)

        MadNLP.print_iter(solver)

        # evaluate termination criteria
        MadNLP.@trace(solver.logger,"Evaluating termination criteria.")
        max(solver.inf_pr,solver.inf_du,solver.inf_compl) <= solver.opt.tol && return MadNLP.SOLVE_SUCCEEDED
        max(solver.inf_pr,solver.inf_du,solver.inf_compl) <= solver.opt.acceptable_tol ?
            (solver.cnt.acceptable_cnt < solver.opt.acceptable_iter ?
            solver.cnt.acceptable_cnt+=1 : return MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL) : (solver.cnt.acceptable_cnt = 0)
        max(solver.inf_pr,solver.inf_du,solver.inf_compl) >= solver.opt.diverging_iterates_tol && return MadNLP.DIVERGING_ITERATES
        solver.cnt.k>=solver.opt.max_iter && return MadNLP.MAXIMUM_ITERATIONS_EXCEEDED
        time()-solver.cnt.start_time>=solver.opt.max_wall_time && return MadNLP.MAXIMUM_WALLTIME_EXCEEDED

        # update the barrier parameter
        MadNLP.@trace(solver.logger,"Updating the barrier parameter.")
        while solver.mu != max(solver.opt.mu_min,solver.opt.tol/10) &&
            max(solver.inf_pr,solver.inf_du,inf_compl_mu) <= solver.opt.barrier_tol_factor*solver.mu
            mu_new = MadNLP.get_mu(solver.mu,solver.opt.mu_min,
                            solver.opt.mu_linear_decrease_factor,solver.opt.mu_superlinear_decrease_power,solver.opt.tol)
            inf_compl_mu = MadNLP.get_inf_compl(solver.x_lr,solver.xl_r,solver.zl_r,solver.xu_r,solver.x_ur,solver.zu_r,solver.mu,sc)
            solver.tau= MadNLP.get_tau(solver.mu,solver.opt.tau_min)
            solver.mu = mu_new
            empty!(solver.filter)
            push!(solver.filter,(solver.theta_max,-Inf))
        end

        #####
        # Newton direction
        #####
        MadNLP.@trace(solver.logger,"Computing the newton step.")

        MadNLP.set_aug_diagonal!(solver.kkt,solver)
        MadNLP.set_aug_rhs!(solver, solver.kkt, solver.c)
        MadNLP.dual_inf_perturbation!(MadNLP.primal(solver.p),solver.ind_llb,solver.ind_uub,solver.mu,solver.opt.kappa_d)

        MadNLP.@trace(solver.logger,"Solving primal-dual system.")
        MadNLP.factorize_wrapper!(solver)
        MadNLP.solve_refine_wrapper!(solver,solver.d,solver.p)
        MadNLP.finish_aug_solve!(solver, solver.kkt, solver.mu)

        #####
        # Line-search
        #####
        MadNLP.@trace(solver.logger,"Backtracking line search initiated.")
        theta = MadNLP.get_theta(solver.c)
        varphi= MadNLP.get_varphi(solver.obj_val,solver.x_lr,solver.xl_r,solver.xu_r,solver.x_ur,solver.mu)
        varphi_d = MadNLP.get_varphi_d(
            MadNLP.primal(solver.f),
            MadNLP.primal(solver.x),
            MadNLP.primal(solver.xl),
            MadNLP.primal(solver.xu),
            MadNLP.primal(solver.d),
            solver.mu,
        )

        alpha_max = MadNLP.get_alpha_max(
            MadNLP.primal(solver.x),
            MadNLP.primal(solver.xl),
            MadNLP.primal(solver.xu),
            MadNLP.primal(solver.d),
            solver.tau,
        )
        solver.alpha_z = MadNLP.get_alpha_z(solver.zl_r,solver.zu_r,MadNLP.dual_lb(solver.d),MadNLP.dual_ub(solver.d),solver.tau)
        alpha_min = MadNLP.get_alpha_min(theta,varphi_d,solver.theta_min,solver.opt.gamma_theta,solver.opt.gamma_phi,
                                  solver.opt.alpha_min_frac,solver.opt.delta,solver.opt.s_theta,solver.opt.s_phi)
        solver.cnt.l = 1
        solver.alpha = alpha_max
        varphi_trial= 0.
        theta_trial = 0.
        small_search_norm = MadNLP.get_rel_search_norm(
            MadNLP.primal(solver.x),
            MadNLP.primal(solver.d),
        ) < 10*eps(T)
        switching_condition = MadNLP.is_switching(varphi_d,solver.alpha,solver.opt.s_phi,solver.opt.delta,2.,solver.opt.s_theta)
        armijo_condition = false
        while true
            copyto!(MadNLP.full(solver.x_trial), MadNLP.full(solver.x))
            axpy!(solver.alpha, MadNLP.primal(solver.d), MadNLP.primal(solver.x_trial))

            solver.obj_val_trial = MadNLP.eval_f_wrapper(solver, solver.x_trial)
            MadNLP.eval_cons_wrapper!(solver, solver.c_trial, solver.x_trial)

            theta_trial = MadNLP.get_theta(solver.c_trial)
            varphi_trial= MadNLP.get_varphi(solver.obj_val_trial,solver.x_trial_lr,solver.xl_r,solver.xu_r,solver.x_trial_ur,solver.mu)
            armijo_condition = MadNLP.is_armijo(varphi_trial,varphi,solver.opt.eta_phi,solver.alpha,varphi_d)

            small_search_norm && break

            solver.ftype = MadNLP.get_ftype(
                solver.filter,theta,theta_trial,varphi,varphi_trial,switching_condition,armijo_condition,
                solver.theta_min,solver.opt.obj_max_inc,solver.opt.gamma_theta,solver.opt.gamma_phi,
                MadNLP.has_constraints(solver))
            if solver.ftype in ["f","h"]
                MadNLP.@trace(solver.logger,"Step accepted with type $(solver.ftype)")
                break
            end

            solver.cnt.l==1 && theta_trial>=theta && MadNLP.second_order_correction(
                solver,alpha_max,theta,varphi,theta_trial,varphi_d,switching_condition) && break

            solver.alpha /= 2
            solver.cnt.l += 1
            if solver.alpha < alpha_min
                MadNLP.@debug(solver.logger,
                       "Cannot find an acceptable step at iteration $(solver.cnt.k). Switching to restoration phase.")
                solver.cnt.k+=1
                return MadNLP.RESTORE
            else
                MadNLP.@trace(solver.logger,"Step rejected; proceed with the next trial step.")
                solver.alpha * norm(MadNLP.primal(solver.d)) < eps(T)*10 &&
                    return solver.cnt.acceptable_cnt >0 ?
                    MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL : MadNLP.SEARCH_DIRECTION_BECOMES_TOO_SMALL
            end
        end

        MadNLP.@trace(solver.logger,"Updating primal-dual variables.")
        copyto!(MadNLP.full(solver.x), MadNLP.full(solver.x_trial))
        copyto!(solver.c, solver.c_trial)
        solver.obj_val = solver.obj_val_trial
        adjusted = MadNLP.adjust_boundary!(solver.x_lr,solver.xl_r,solver.x_ur,solver.xu_r,solver.mu)
        adjusted > 0 &&
            MadNLP.@warn(solver.logger,"In iteration $(solver.cnt.k), $adjusted Slack too small, adjusting variable bound")

        axpy!(solver.alpha, MadNLP.dual(solver.d), solver.y)

        solver.zl_r .+= solver.alpha_z .* MadNLP.dual_lb(solver.d)
        solver.zu_r .+= solver.alpha_z .* MadNLP.dual_ub(solver.d)

        MadNLP.reset_bound_dual!(
            MadNLP.primal(solver.zl),
            MadNLP.primal(solver.x),
            MadNLP.primal(solver.xl),
            solver.mu,solver.opt.kappa_sigma,
        )
        MadNLP.reset_bound_dual!(
            MadNLP.primal(solver.zu),
            MadNLP.primal(solver.xu),
            MadNLP.primal(solver.x),
            solver.mu,solver.opt.kappa_sigma,
        )
        MadNLP.eval_grad_f_wrapper!(solver, solver.f, solver.x)

        if !switching_condition || !armijo_condition
            MadNLP.@trace(solver.logger,"Augmenting filter.")
            MadNLP.augment_filter!(solver.filter,theta_trial,varphi_trial,solver.opt.gamma_theta)
        end

        solver.cnt.k+=1
        MadNLP.@trace(solver.logger,"Proceeding to the next interior point iteration.")
    end
end

function l_solve!(
    solver::MadNLP.AbstractMadNLPSolver;
    kwargs...
)
    stats = MadNLP.MadNLPExecutionStats(solver)
    nlp = solver.nlp
    solver.cnt.start_time = time()

    if !isempty(kwargs)
        @warn(solver.logger,"The options set during resolve may not have an effect")
        MadNLP.set_options!(solver.opt, kwargs)
    end

    try
        MadNLP.@notice(solver.logger,"This is MadLP, running with $(MadNLP.introduce(solver.linear_solver))\n")
        MadNLP.print_init(solver)
        MadNLP.initialize!(solver)
        solver.status = mpc!(solver)
    catch e
        if e isa MadNLP.InvalidNumberException
            if e.callback == :obj
                solver.status=MadNLP.INVALID_NUMBER_OBJECTIVE
            elseif e.callback == :grad
                solver.status=MadNLP.INVALID_NUMBER_GRADIENT
            elseif e.callback == :cons
                solver.status=MadNLP.INVALID_NUMBER_CONSTRAINTS
            elseif e.callback == :jac
                solver.status=MadNLP.INVALID_NUMBER_JACOBIAN
            elseif e.callback == :hess
                solver.status=MadNLP.INVALID_NUMBER_HESSIAN_LAGRANGIAN
            else
                solver.status=MadNLP.INVALID_NUMBER_DETECTED
            end
        elseif e isa MadNLP.NotEnoughDegreesOfFreedomException
            solver.status=MadNLP.NOT_ENOUGH_DEGREES_OF_FREEDOM
        elseif e isa MadNLP.LinearSolverException
            solver.status=MadNLP.ERROR_IN_STEP_COMPUTATION;
            solver.opt.rethrow_error && rethrow(e)
        elseif e isa InterruptException
            solver.status=MadNLP.USER_REQUESTED_STOP
            solver.opt.rethrow_error && rethrow(e)
        else
            solver.status=MadNLP.INTERNAL_ERROR
            solver.opt.rethrow_error && rethrow(e)
        end
    finally
        solver.cnt.total_time = time() - solver.cnt.start_time
        !(solver.status < MadNLP.SOLVE_SUCCEEDED) && (MadNLP.print_summary_1(solver);MadNLP.print_summary_2(solver))
        # Unscale once the summary has been printed out
        MadNLP.unscale!(solver)
        MadNLP.finalize(solver.logger)

        MadNLP.update!(stats,solver)
    end

    return stats
end
