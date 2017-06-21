require 'json'
require 'matrix'
require './enumerable_extension'
# require './matrix_extension'

module LinearRegression
    class LinearRegressionBase
        def train(entries)
            raise "hasn't implemented"
        end

        def beta
            @beta
        end

        def predict(vector)
            x = Matrix[vector + [1]]
            x_cross_beta = x * @beta
            return x_cross_beta[0,0]
        end

        def r_squared_score(new_entries)
            y = new_entries.map{|v|v[:label]}
            y_bar = y.mean
            ss_tot = 0.0
            y.each do |v|
                ss_tot += (v-y_bar)**2
            end
            # puts "ss_tot=#{ss_tot}"

            ss_res = 0.0
            new_entries.each do |e|
                prediction = predict(e[:features])
                ss_res += (e[:label]-prediction)**2
            end
            #puts "ss_res=#{ss_res}"
            return 1-(ss_res/ss_tot)
        end
    end

    class ALS < LinearRegressionBase
        def train(entries)
            label = entries.map{|e| e[:label] }
            features = entries.map{|e| e[:features] }.map{|e| e+[1] }

            @dimension = features[0].size

            y = Matrix[label].t
            x = Matrix[*features]

            @beta = (x.t * x).inv * x.t * y
        end
    end

    class GD < LinearRegressionBase
        def initialize(numIter: 100, alpha: 0.01)
            @numIter = numIter
            @alpha = alpha
        end

        def train(entries, &block)
            label = entries.map{|e| e[:label] }
            features = entries.map{|e| e[:features] }.map{|e| e+[1]}

            @dimension = features[0].size

            y = Matrix[label].t
            x = Matrix[*features]

            @beta = Matrix[@dimension.times.map{|x|1}].t

            @numIter.times do |it|
                y_bar = x * @beta
                loss = y_bar - y
                gradient = (x.t * loss)/ entries.size
                @beta = @beta - (@alpha * gradient)
                yield it,@beta,loss if block!=nil
            end
        end
    end
end
