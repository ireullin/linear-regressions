require './linear_regressions'

trainee = [
    {features: [6,2],  label: 7},
    {features: [8,1],  label: 9},
    {features: [10,0], label: 13},
    {features: [14,2], label: 17.5},
    {features: [18,0], label: 18}
]

test = [
    {features: [8,2],  label: 11},
    {features: [9,0],  label: 8.5},
    {features: [11,2], label: 15},
    {features: [16,2], label: 18},
    {features: [12,0], label: 11}
]

als = LinearRegression::ALS.new
als.train(trainee)
puts als.beta
puts als.r_squared_score(test)

gd = LinearRegression::GD.new(numIter: 1000, alpha: 0.01)
gd.train(trainee)
puts gd.beta
puts gd.r_squared_score(test)
