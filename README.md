# principal-components-analysis

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'principal-components-analysis'
```

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install principal-components-analysis

## Usage

A simple example:

```ruby
require 'principal-components-analysis'

data = [[2,   0,   -1.4],
        [2.2, 0.2, -1.5],
        [2.4, 0.1, -1],
        [1.9, 0,   -1.2]]

# create a pca object within data
pca = PCA.new(data)

# show the eigen values & eigen vectors for the data
puts "#{pca.eigen}\n\n"

# reduce to 2 dimensions
reduced_data, distortion_rate = pca.reduce(2)
p reduced_data
puts "distortion_rate=#{distortion_rate}\n\n"

# reduce to 1 dimensions
reduced_data, distortion_rate = pca.reduce(1)
p reduced_data
puts "distortion_rate=#{distortion_rate}\n\n"
```

Output below:

```
[{:value=>0.06896846369330273, :vector=>[-0.25852548789080637, -0.6910187907758658, 0.6750241498683445]}, {:value=>0.036928271336293624, :vector=>[-0.892351724749828, 0.4384535489558841, 0.10708354095750225]}, {:value=>0.001603264970403666, :vector=>[0.3699634731251552, 0.5746751397116905, 0.7299832274446598]}]

Matrix[[-1.462084785597295, -1.9346204068401591],
 [-1.719496056317464,   -2.0361083960946984],
 [-1.3645871998838663,  -2.204882325461501],
 [-1.3012274068345455,  -1.8239685261736758]]
distortion_rate=0.014914092747941088

Matrix[[-1.462084785597295],
 [-1.719496056317464],
 [-1.3645871998838663],
 [-1.3012274068345455]]
distortion_rate=0.3584328958762537
```
