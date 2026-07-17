@testitem "Package loads" begin
    using DistributionsInference

    @test isdefined(DistributionsInference, :DistributionsInference)
end
