# coding: utf-8
lib = File.expand_path('../lib', __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)

Gem::Specification.new do |spec|
  spec.name          = "principal-components-analysis"
  spec.version       = '0.0.1'
  spec.authors       = ["ireullin"]
  spec.email         = ["ireullin@gmail.com"]
  spec.date          = '2017-06-02'
  spec.homepage      = 'https://github.com/ireullin/principal-components-analysis'
  spec.summary       = %Q{A PCA algorithm for reducing dimension}
  spec.description   = %Q{A PCA algorithm for reducing dimension}
  spec.license       = "MIT"
  spec.files         = [
    'lib/principal-components-analysis.rb',
    'lib/enumerable_extension.rb',
    'lib/matrix_extension.rb']
end
