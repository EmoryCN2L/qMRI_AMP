classdef A_wav_fa_3d_LinTrans < LinTrans
	% A_wav_fa_3d_LinTrans:  Linear transform class for function handles. 

	properties
		M	% output dimension: M=sm_num*echo_num*coil_num
		N	% input dimension: N=mat_sz(1)*mat_sz(2)*echo_num
		mat_sz	% the size of the 2D echo images
		wav_coef_len	% the first dimensionality of y, the length of wavelet coefficient
		echo_num	% the dictionary dimension
        fa_num      % the number of flip angles
		A	% function handle for forward multiply
		Ah	% function handle for hermition-transpose multiply
        E   % construct a 3d wavelet structure
        Et  % extract the 3d wavelet coefficients
		S	% function handle for forward multiply-square
		St	% function handle for transpose multiply-square
		%FrobNorm    % 1/(M*N) times the squared Frobenius norm 
	end

	methods

		% Constructor
		function obj = A_wav_fa_3d_LinTrans(M,N,mat_sz,wav_coef_len,echo_num,fa_num,A,Ah,E,Et,S,St)
			obj = obj@LinTrans;

			% manditory inputs
			if ~(isa(N,'numeric')&isa(M,'numeric'))
				error('First and second inputs must be integers')   
			end
			obj.M = M;
			obj.N = N;
			obj.mat_sz = mat_sz;
			obj.wav_coef_len = wav_coef_len;
			obj.echo_num = echo_num;
            obj.fa_num = fa_num;
			if ~(isa(A,'function_handle')&isa(Ah,'function_handle'))
				error('Third and fourth inputs must be function handles')   
			end
			obj.Ah = Ah;
			obj.A = A;
            obj.E = E;
            obj.Et = Et;

			% optional inputs 
			if nargin > 10
				if isa(S,'double')&&(S>0)
					% 10th input "S" contains FrobNorm
					obj.FrobNorm = S;
				elseif (nargin > 10)&&(isa(S,'function_handle')&isa(St,'function_handle'))
					% 10th and 11th inputs are both function handles, S and St
					obj.S = S;
					obj.St = St;
				else
					error('Problem with the 8th & 9th inputs.  We need that either the fifth input is a positive number for FrobNorm, or that the fifth and sixth inputs are both function handles for S and St.')   
				end
			else
				% approximate the squared Frobenius norm
				P = 2;      % increase for a better approximation
				obj.FrobNorm = 0;
				for p=1:P,   % use "for" since A may not support matrices 
			
					x_tmp = randn([wav_coef_len echo_num fa_num]) ;
					norm_x_tmp = norm(x_tmp(:),'fro');
					y_tmp = obj.mult(x_tmp);
					obj.FrobNorm = obj.FrobNorm + (norm(y_tmp(:), 'fro')/norm_x_tmp).^2;
                    fprintf('%d\t%d\n',p,obj.FrobNorm)
				end
				obj.FrobNorm = sqrt(obj.FrobNorm * (N/P));
			end
			
			if (M~=mat_sz(1)*mat_sz(2)*mat_sz(3)*echo_num*fa_num)
				error('Dimension mismatch: M')
			end
			
			if (N~=wav_coef_len*echo_num*fa_num)
				error('Dimension mismatch: N')
			end
		end
		
		% Size
		function [m,n] = size(obj,dim)
			if nargin>1 % a specific dimension was requested
				if dim==1
					m=obj.M;
				elseif dim==2
					m=obj.N;
				elseif dim>2
					m=1; 
				else
					error('invalid dimension')
				end
			elseif nargout<2  % all dims in one output vector
				m=[obj.M,obj.N];
			else % individual outputs for the dimensions
				m = obj.M;
				n = obj.N;
			end
		end

		% Matrix multiply
		function y = mult(obj,x)
			y=zeros([obj.mat_sz obj.echo_num obj.fa_num]);
	
            for (f=1:obj.fa_num)
			for (j=1:obj.echo_num)
				y(:,:,:,j,f) = obj.A(obj.E(x(:,j,f)));
			end
            end

		end

		% Hermitian-transposed-Matrix multiply 
		function x = multTr(obj,y)
			x = zeros([obj.wav_coef_len obj.echo_num obj.fa_num]);

            for (f=1:obj.fa_num)
			for (j=1:obj.echo_num)
				x(:,j,f) = obj.Et(obj.Ah(y(:,:,:,j,f)));
			end
            end
		end


		% Squared-Matrix multiply 
		function y = multSq(obj,x)
			if isempty(obj.FrobNorm)
				y = obj.S(x);
			else
				%y = obj.echo_num*obj.fa_num*ones([obj.mat_sz obj.echo_num obj.fa_num])*((obj.FrobNorm^2/(obj.M*obj.N))*sum(x,'all'));
                y = obj.echo_num * obj.fa_num * (obj.FrobNorm^2/obj.M*x);
			end
		end


		% Squared-Hermitian-Transposed Matrix multiply 
		function x = multSqTr(obj,y)
			if isempty(obj.FrobNorm)
				x = obj.St(y);
			else
				%x = obj.echo_num*obj.fa_num*ones([obj.wav_coef_len obj.echo_num obj.fa_num])*((obj.FrobNorm^2/(obj.M*obj.N))*sum(y,'all'));
                x = obj.echo_num * obj.fa_num * (obj.FrobNorm^2/obj.N*y);
			end
		end

	end
end
