function main()
file = open("shake.top", "r")

shake_data = readlines(file) .|> split
shake_data = permutedims(hcat(shake_data[2:end]...))

start   = 353
n_atoms = 22
n_molecules = 12

top1 = [357, 363, 367, 370]
top1_mod = top1 .- start

top2 = [359, 365]
top2_mod = top2 .- start

average1 = 0.0
average2 = 0.0

output1 = Vector()
output2 = Vector()

for i in 1:n_molecules

    top_now1 = top1_mod .+ (i-1)*n_atoms .+ start
    top_now2 = top2_mod .+ (i-1)*n_atoms .+ start

    for j in 1:length(shake_data[:,1])

        if(shake_data[j,1] ∈  string.(top_now1))

            average1 += parse(Float64, shake_data[j,3])
            push!(output1, string(shake_data[j,1], "   ", shake_data[j,2]))

        elseif(shake_data[j,1] ∈  string.(top_now2))

            average2 += parse(Float64, shake_data[j,3])
            push!(output2, string(shake_data[j,1], "   ", shake_data[j,2]))

        end
    end
end

for i in output1
    println(i * "  " * string(average1 / (n_molecules * length(top1))))
end
for i in output2
    println(i * "  " * string(average2 / (n_molecules * length(top2))))
end

end

main()