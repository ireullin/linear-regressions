# coding: utf-8
lib = File.expand_path('../lib', __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)

Gem::Specification.new do |spec|
  spec.name          = "linear-regressions"
  spec.version       = '0.0.1'
  spec.authors       = ["ireullin"]
  spec.email         = ["ireullin@gmail.com"]
  spec.date          = '2017-06-21'
  spec.homepage      = 'https://github.com/ireullin/linear-regressions'
  spec.summary       = %Q{Linear regression algorithms which implemented Alternating Least Squares & Gradient Descent}
  spec.description   = %Q{Linear regression algorithms which implemented Alternating Least Squares & Gradient Descent}
  spec.license       = "MIT"
  spec.files         = [
    'lib/linear_regressions.rb',
    'lib/enumerable_extension.rb',
    'lib/matrix_extension.rb']
end
