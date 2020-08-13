using KernelAbstractions, Test, CUDA

if has_cuda_gpu()
    CUDA.allowscalar(false)
end

# Simple kernel for matrix multiplication
@kernel function vadd_kernel!(c, a, b)
    i, j = @index(Global, NTuple)

    c[i,j] = a[i,j] + b[i,j]
end

# Creating a wrapper kernel for launching with error checks
function vadd!(a, b, c)
    if isa(a, Array)
        kernel! = vadd_kernel!(CPU(),4)
    else
        kernel! = vadd_kernel!(CUDADevice(),256)
    end
    kernel!(c, a, b, ndrange=size(c)) 
end

a = ones(1024, 1024)
b = ones(1024, 1024)
c = zeros(1024, 1024)

# beginning CPU tests, returns event
ev = vadd!(a,b,c)
wait(ev)

@test isapprox(c, a.+b)

# beginning GPU tests
d_a = CuArray(a)
d_b = CuArray(b)
d_c = CuArray(c)

ev = vadd!(d_a, d_b, d_c)
wait(ev)

@test isapprox(Array(d_c), a.+b)
