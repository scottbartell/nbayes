require 'redis'

# == NBayes::Base
#
# Robust implementation of NaiveBayes:
# - using log probabilities to avoid floating point issues
# - Laplacian smoothing for unseen tokens
# - allows binarized or standard NB
# - allows Prior distribution on category to be assumed uniform (optional)
# - generic to work with all types of tokens, not just text


module NBayes

  class Vocab
    attr_accessor :log_size, :redis, :prefix

    def initialize(log_size:, redis: Redis.new, prefix: "nbayes:vocab:")
      @redis = redis
      @prefix = prefix

      # for smoothing, use log of vocab size, rather than vocab size
      @log_size = log_size
    end

    def tokens
      redis.hkeys(prefix + "tokens")
    end

    def delete(token)
      redis.hdel(prefix + "tokens", token)
    end

    def each(&block)
      tokens.each(&block)
    end

    def size
      count = redis.hlen(prefix + "tokens")
      if log_size
        Math.log(count)
      else
        count
      end
    end

    def seen_token(token)
      redis.hincrby(prefix + "tokens", token, 1)
    end
  end

  class Data
    attr_reader :redis, :prefix

    CATEGORIES_KEY = "categories"

    def initialize(redis: Redis.new, prefix: "nbayes:data:")
      @redis = redis
      @prefix = prefix
    end

    def categories
      redis.hkeys(prefix + CATEGORIES_KEY)
    end

    # UNCHANGED
    def each(&block)
      categories.each(&block)
    end

    def token_trained?(token, category)
      redis.hexists(prefix + category, token)
    end

    def increment_examples(category)
      redis.hincrby(prefix + CATEGORIES_KEY, category, 1)
    end

    def decrement_examples(category)
      updated_example_count = redis.hincrby(prefix + CATEGORIES_KEY, category, -1)
      if updated_example_count < 1
        delete_category(category)
      end
      updated_example_count
    end

    def example_count(category)
      redis.hget(prefix + CATEGORIES_KEY, category).to_i
    end

    def token_count(category)
      redis.hvals(prefix + category).inject(0) {|sum, x| sum + x.to_i}
    end

    def total_examples
      redis.hvals(prefix + CATEGORIES_KEY).inject(0) {|sum, x| sum + x.to_i}
    end

    def add_token_to_category(category, token)
      redis.hincrby(prefix + category, token, 1)
    end

    def remove_token_from_category(category, token)
      updated_token_count = redis.hincrby(prefix + category, token, -1)
      if updated_token_count < 1
        delete_token_from_category(category, token)
      end
      updated_token_count
    end

    def count_of_token_in_category(category, token)
      redis.hget(prefix + category, token).to_i
    end

    def delete_token_from_category(category, token)
      number_fields_deleted = redis.hdel(prefix + category, token)
      number_fields_deleted != 0
    end

    # UNCHANGED
    def purge_less_than(token, x)
      return if token_count_across_categories(token) >= x
      categories.each do |category|
        delete_token_from_category(category, token)
      end
      true  # Let caller know we removed this token
    end

    # UNCHANGED
    def token_count_across_categories(token)
      categories.inject(0) do |sum, category|
        sum + count_of_token_in_category(category, token)
      end
    end

    def delete_category(category)
      redis.hdel(prefix + CATEGORIES_KEY, category)
      categories
    end
  end

  class Base

    attr_accessor :assume_uniform, :debug, :k, :vocab, :data
    attr_reader :binarized

    def initialize(redis: Redis.new, binarized: false, log_vocab: nil)
      @debug = false
      @k = 1
      @binarized = binarized
      @assume_uniform = false
      @vocab = Vocab.new(:log_size => log_vocab, :redis => redis)
      @data = Data.new(:redis => redis)
    end

    # Allows removal of low frequency words that increase processing time and may overfit
    # - tokens with a count less than x (measured by summing across all classes) are removed
    # Ex: nb.purge_less_than(2)
    #
    # NOTE: this does not decrement the "examples" count, so purging is not *always* the same
    # as if the item was never added in the first place, but usually so
    def purge_less_than(x)
      remove_list = {}
      @vocab.each do |token|
        if data.purge_less_than(token, x)
          # print "removing #{token}\n"
          remove_list[token] = 1
        end
      end  # each vocab word
      remove_list.keys.each {|token| @vocab.delete(token) }
      # print "total vocab size is now #{vocab.size}\n"
    end

    # Delete an entire category from the classification data
    def delete_category(category)
      data.delete_category(category)
    end

    def train(tokens, category)
      tokens = tokens.uniq if binarized
      data.increment_examples(category)
      tokens.each do |token|
        vocab.seen_token(token)
        data.add_token_to_category(category, token)
      end
    end

    # Be careful with this function:
    # * It decrements the number of examples for the category.
    #   If the being-untrained category has no more examples, it is removed from the category list.
    # * It untrains already trained tokens, non existing tokens are not considered.
    def untrain(tokens, category)
      tokens = tokens.uniq if binarized
      data.decrement_examples(category)
      
      tokens.each do |token|
        if data.token_trained?(token, category)
          vocab.delete(token)
          data.remove_token_from_category(category, token)
        end
      end
    end

    def classify(tokens)
      print "classify: #{tokens.join(', ')}\n" if @debug
      probs = {}
      tokens = tokens.uniq if binarized
      probs = calculate_probabilities(tokens)
      print "results: #{probs.to_yaml}\n" if @debug
      probs.extend(NBayes::Result)
      probs
    end

    def category_stats
      data.category_stats
    end

    # Calculates the actual probability of a class given the tokens
    # (this is the work horse of the code)
    def calculate_probabilities(tokens)
      # P(class|words) = P(w1,...,wn|class) * P(class) / P(w1,...,wn)
      #                = argmax P(w1,...,wn|class) * P(class)
      #
      # P(wi|class) = (count(wi, class) + k)/(count(w,class) + kV)
      prob_numerator = {}
      v_size = vocab.size

      cat_prob = Math.log(1 / data.categories.count.to_f)
      total_example_count = data.total_examples.to_f

      data.each do |category|
        unless assume_uniform
          cat_prob = Math.log(data.example_count(category) / total_example_count)
        end

        log_probs = 0
        denominator = (data.token_count(category) + @k * v_size).to_f
        tokens.each do |token|
          numerator = data.count_of_token_in_category(category, token) + @k
          log_probs += Math.log( numerator / denominator )
        end
        prob_numerator[category] = log_probs + cat_prob
      end
      normalize(prob_numerator)
    end

    def normalize(prob_numerator)
      # calculate the denominator, which normalizes this into a probability; it's just the sum of all numerators from above
      normalizer = 0
      prob_numerator.each {|cat, numerator| normalizer += numerator }
      # One more caveat:
      # We're using log probabilities, so the numbers are negative and the smallest negative number is actually the largest prob.
      # To convert, we need to maintain the relative distance between all of the probabilities:
      # - divide log prob by normalizer: this keeps ratios the same, but reverses the ordering
      # - re-normalize based off new counts
      # - final calculation
      # Ex: -1,-1,-2  =>  -4/-1, -4/-1, -4/-2
      #   - renormalize and calculate => 4/10, 4/10, 2/10
      intermed = {}
      renormalizer = 0
      prob_numerator.each do |cat, numerator|
        intermed[cat] = normalizer / numerator.to_f
        renormalizer += intermed[cat]
      end
      # calculate final probs
      final_probs = {}
      intermed.each do |cat, value|
        final_probs[cat] = value / renormalizer.to_f
      end
      final_probs
    end

  end

  module Result
    # Return the key having the largest value
    def max_class
      keys.max{ |a,b| self[a] <=> self[b] }
    end
  end

end
