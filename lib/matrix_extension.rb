
class Matrix
	def covariance_matrix
		dim = self.column_size
		buff = Array.new(dim){Array.new(dim,0)}
		0.upto(dim-1) do |i|
			i.upto(dim-1) do |j|
				if i==j
					buff[i][j] = self.column(i).to_a.variance
				else
					conv = self.column(i).to_a.covariance(self.column(j).to_a)
					buff[i][j] = conv
					buff[j][i] = conv
				end
			end
		end
		Matrix[*buff]
	end

    def l1_normalize
        buff = Array.new
        self.row_size.times do |i|
            buff << self.row(i).to_a.l1_normalize
        end
        Matrix[*buff]
    end

    def l2_normalize
        buff = Array.new
        self.row_size.times do |i|
            buff << self.row(i).to_a.l2_normalize
        end
        Matrix[*buff]
    end

	def to_json(*param)
		buff = []
		self.row_size.times do |i|
            buff << self.row(i).to_a
		end
		return buff.to_json(param)
	end

    def inspect
        buff = ""
        self.row_size.times do |i|
            if i==0
                buff += "Matrix["
            else
                buff += " "
            end

            buff += "[" + self.row(i).to_a.join(",\t") + "]"

            if i==self.row_size-1
                buff += "]"
            else
                buff += ",\n"
            end
        end
        return buff
    end

	alias to_s inspect
end
