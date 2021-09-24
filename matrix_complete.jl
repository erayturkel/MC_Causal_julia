##==================================================
##
## Script name: Causal Matrix Completion
##
##
## Purpose of script: Given a Matrix of observed Y(0), indicator for missing
## entries and parameters for the algorithm, run the causal matrix Completion
## method as described in Athey et al. (JASA 2021)
## Author: Eray Turkel
##
## Date Last Edited: 09-22-2021
##
## Email: eturkel@stanford.edu
##
## Input:  TBC



using LinearAlgebra: pinv, svd, norm, I, dot, diagm,Diagonal, QRIteration
using Arpack: svds
using Statistics: mean
using Random

function shrink_matrix(A, lambda)
    # Implements the matrix shrinkage operator.
    # Denoting the SVD of A= U S V',
    # Replaces the matrix of eigenvalues S with S_shrink
    # where S_shrink replaces the i'th singular value
    # in the matrix S with max(sigma_i(A)- lambda, 0). 
    N = size(A)[1];
    T = size(A)[2];
    rank_max=min(N,T);
    F = svd(A);
    S = F.S;
    num_singular=size(S)[1];
    for i in 1:num_singular
        S[i]=max(S[i]-lambda,0);
    end
    return  F.U* Diagonal(S) * F.Vt;
end

function FEestimator(Y, L, O)
    #Extract the time and unit fixed effects
    #Given Y, current estimate of L, and Observed Matrix O
    est_matrix=( (Y-L) .* O) ;
    N=size(Y)[1];
    T=size(Y)[2];
    time_columns=[ceil(i/N)==j for i in 1:(N*T), j in 1:T];
    unit_columns=[(i%N)==(j%N) for i in 1:(N*T), j in 1:N];
    X=hcat(time_columns,unit_columns);
    X=X[(vec(O).!=0),:];
    y=vec(est_matrix)[(vec(O).!=0),];
    fixed_effects=inv(X' * X) * (X' * vec(y));
    time_FE= fixed_effects[1:T]
    unit_FE= fixed_effects[(T+1):(T+N)]
    return (T=time_FE, U=unit_FE);
end


function softImpute_noFE(Y,O,lambda, n_iter,verbose= true)
    # Y is the Y(0) matrix of size N by T
    # O is the matrix of size N by T marking observed entries,
    # with O_(i,t)=1 if Y(0)_(i,t) is observed, O_(i,t)=0 otherwise.
    # lambda is the regularization parameter
    N=size(Y)[1];
    T=size(Y)[2];
    N_Obs=(sum(sum(O,dims=2),dims=1))[1];
    O_orth=((O .- 1) .* (-1));
    P_o_Y=(Y .* O);
    L_prev= P_o_Y;
    L_next= shrink_matrix(((Y) .* O)+ (L_prev .* O_orth), lambda*(N_Obs/2));
    for i in 1:n_iter
        L_prev=deepcopy(L_next)
        L_next= shrink_matrix(((Y) .* O)+ (L_prev .* O_orth), lambda*(N_Obs/2));
        distance= norm((L_next-L_prev));
        if (verbose)
            println("Current Dist:");
            println(distance);
        end
    end
    return L_next
end


function softImpute(Y,O,lambda, n_iter, verbose= true)
    # Y is the Y(0) matrix of size N by T
    # O is the matrix of size N by T marking observed entries,
    # with O_(i,t)=1 if Y(0)_(i,t) is observed, O_(i,t)=0 otherwise.
    # lambda is the regularization parameter
    N=size(Y)[1];
    T=size(Y)[2];
    N_Obs=(sum(sum(O,dims=2),dims=1))[1];
    O_orth=((O .- 1) .* (-1));
    P_o_Y=(Y .* O);
    L_prev= P_o_Y;
    fixed_effects= FEestimator(Y,zeros(N,T),O);
    Time= ones(N,1) * (fixed_effects.T)';
    Unit= (fixed_effects.U) * ones(1,T);
    L_next= shrink_matrix(((Y-Time-Unit) .* O)+ (L_prev .* O_orth), lambda*(N_Obs/2));
    for i in 1:n_iter
        L_prev= deepcopy(L_next);
        fixed_effects= FEestimator(Y,L_prev,O);
        Time= ones(N,1) * (fixed_effects.T)';
        Unit= (fixed_effects.U) * ones(1,T);
        L_new= shrink_matrix(((Y-Time-Unit) .* O)+ ((L_prev) .* O_orth), lambda*(N_Obs/2));
        L_next= deepcopy(L_new);
        distance= norm((L_next-L_prev));
        if (verbose)
            println("Current Dist:");
            println(distance);
        end
    end
    return (L=L_next, T=Time, U=Unit, Y_est= L_next+ Time+ Unit)
end


function MC_estimator(Y,O,reg_lambda, n_iter, FE_estimate= false, verbose=true)
    if (FE_estimate)
        return softImpute(Y,O,reg_lambda, n_iter, verbose);
    else 
        return softImpute_noFE(Y,O,reg_lambda, n_iter, verbose);
    end
end


unit_fixed_rand= 8   .+ randn(1000)*0.5;
time_fixed_rand= 0.5 .+ randn(50)*0.1;
#Create random low rank matrix for the true dgp
L_rand = randn(1000,50)*0.5;
L_low = svd(L_rand);
Sigmas= L_low.S;
num_sigmas=size(Sigmas)[1];
Sigmas[(num_sigmas):num_sigmas]=zeros(num_sigmas-(num_sigmas)+1);
L_rand=  L_low.U* Diagonal(Sigmas) * L_low.V;
eps = randn(1000,50)*0.05;

Y= unit_fixed_rand * ones(1,50) + ones(1000,1) * time_fixed_rand' + L_rand +eps;

N=size(Y)[1];
T=size(Y)[2];

O_1=[ j<15 for i in 1:(150), j in 1:T];
O_2=[ j<25 for i in 1:(150), j in 1:T];
O_3=[ j<35 for i in 1:(150), j in 1:T];
O_4=[ j<(T+1) for i in 1:(550), j in 1:T];

O=vcat(O_1,O_2,O_3,O_4);


(MC_estimator(Y,O,0.001,1000))


(MC_estimator(Y,O,0.001,50,true))
