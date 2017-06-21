# linear-regressions

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'linear-regressions'
```

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install linear-regressions

## Usage

A simple example:

```ruby
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

# use Alternating Least Squares
als = LinearRegression::ALS.new
als.train(trainee)
puts als.beta
puts als.r_squared_score(test)

# use Gradient Descent
gd = LinearRegression::GD.new(num_iter: 1000, alpha: 0.01)
gd.train(trainee)
puts gd.beta
puts gd.r_squared_score(test)

gd.train(trainee) do |i,beta,loss|
    puts "idexe=#{i} beta=#{beta} loss=#{loss}"
end
```

Output below:

```
Matrix[[1.0104166666666667], [0.3958333333333335], [1.1875]]
0.7701677731318466
Matrix[[1.0200094830556161], [0.4315547941319965], [1.033605114003101]]
0.7783524932121337
idexe=0 beta=Matrix[[0.978], [0.992], [0.997]] loss=Matrix[[2], [1], [-2], [-0.5], [1]]
idexe=1 etc...
```
