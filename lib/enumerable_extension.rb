
module Enumerable
    def entropy
        dataset = Hash.new(0)
        self.each{|x| dataset[x] += 1 }

        entropy = 0.0
        dataset.each do |k,v|
            p = v.to_f / self.size
            entropy += (-p)*Math.log2(p)
        end

        return entropy
    end

    def concitional_entropy_with(label)
        dataset = Hash.new{|h,k| h[k] = Array.new }
        self.each_with_index{|v,i| dataset[v] << label[i] }

        new_entropy = 0.0
        dataset.each{|k,v| new_entropy += (v.size.to_f / self.size)*v.entropy }
        return new_entropy
    end

    def sum
      self.inject(0){|accum, i| accum + i }
    end

    def mean
      self.sum / self.length.to_f
    end

    def geo_mean
        geo_sum = self.inject(0){|accum, i| accum + i*i }
        Math.sqrt(geo_sum)
    end

    def median
        sorted = self.sort
        m = sorted.length / 2
        if sorted.length.odd?
            sorted[m]
        else
            (sorted[m-1]+sorted[m])/2.0
        end
    end

    def sum_and_mean
        sum = self.sum
        mean = sum/self.length.to_f
        return sum,mean
    end

    def variance(ddof=1)
        m = self.mean
        sum = self.inject(0){|accum, i| accum +(i-m)**2 }
        sum / (self.length - ddof).to_f
    end

    def stdev(ddof=1)
        return Math.sqrt(self.variance(ddof))
    end

    def variance_and_stdev(ddof=1)
        sv = self.variance(ddof)
        stdev = Math.sqrt(sv)
        return sv, stdev
    end

    def covariance(arr,ddof=1)
        raise "array length error" if arr.length!=self.length
        xbar = self.mean
        ybar = arr.mean

        accum = 0.0
        arr.length.times do |i|
            accum += (self[i]-xbar)*(arr[i]-ybar)
        end
        return accum / (self.length - ddof).to_f
    end

    def pearson(arr,ddof=1)
        self.covariance(arr,ddof) / (self.stdev(ddof) * arr.stdev(ddof))
    end

    def l1_normalize
        m = self.sum.to_f
        self.map{|x| x / m }
    end

    def l2_normalize
        m = self.geo_mean
        self.map{|x| x / m }
    end
end

