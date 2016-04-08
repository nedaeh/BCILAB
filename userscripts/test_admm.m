
% admm_inputs.mat & admm_outputs.mat
% proper input and output files are generated based on the flag set
%
% fatflag:
%    for n<m, the additional instructions modify the input parameters and
%    generate the corrsponding output file
%
% iterativeflag:
%    
%
%  rho_updateflag:
%                   
%  lambda_updateflag:
%

%     eqblksize
%     compute_objval

load('/Users/Neda.Qusp/Qusp/Projects/SIFTFunctions/ADMM/Sim/admm_inputs_ck.mat');
fatflag = 0;  % for n<m
iterativeflag = 0; %solver method direct or iterative
rho_updateflag = 0;
lambda_updateflag = 1;

for i=1:10
    if fatflag
        A = admm_inputs{i}{2};
        A= A';
        admm_inputs{i}{2} = A(:,1:63);
        
        y = admm_inputs{i}{4};
        y = [y;y;y];
        admm_inputs{i}{4} = y(1:1500);
        
        designMatrixBlockSize = admm_inputs{i}{13};
        admm_inputs{i}{13} = [150,63];
        
        z_init = admm_inputs{i}{9};
        admm_inputs{i}{9} = z_init(1:630);
        
        u_init = admm_inputs{i}{11};
        admm_inputs{i}{11} = u_init(1:630);
        
        blks= 7*ones(1,90);
        admm_inputs{i}{6} = blks;
        
    elseif iterativeflag
        admm_inputs{i}{7}.x_update.arg_selection ='iterative';
    elseif rho_updateflag
        admm_inputs{i}{7}.rho_update = 1;
    elseif lambda_updateflag
        admm_inputs{i}{7}.lambda_update = 1;
    end
    
    [z, u] = admm_gl(admm_inputs{i}{:});
    
    out{i}{1} = z;
    out{i}{2} = u;
end

save('/Users/Neda.Qusp/Qusp/Projects/SIFTFunctions/ADMM/Sim/admm_inputs.mat','admm_inputs')
save('/Users/Neda.Qusp/Qusp/Projects/SIFTFunctions/ADMM/Sim/admm_outputs.mat','out');
