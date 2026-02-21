from math import sqrt, ceil
from itertools import takewhile
from time import perf_counter
import argparse

verbose = False

def text_to_value_list(text: str):
    "Convert a string to a list of ints (character values)"
    return list(map(ord, text))

def signed_value_to_bf(v):
    return '+' * v if v >= 0 else '-' * (-v)

def value_to_bf_const(v):
    return signed_value_to_bf(v) if v < 128 else signed_value_to_bf(v - 256)

def value_to_bf_mult(v):
    if v == 0:
        raise Exception('Cannot convert null byte')
    f1 = int(sqrt(v))
    f2 = v // f1
    rem = v - f1 * f2
    rem2 = abs(v - f1 * (f2 + 1))

    if rem <= rem2 + 1:
        return f'>{"+" * f1}[-<{"+" * f2}>]<{"+" * rem}'

    return f'>{"+" * f1}[-<+{"+" * f2}>]<{"-" * rem2}'

def value_to_bf_optimal(v):
    # for values below 16 sequence of pluses is the same or often more efficient than multiplication-based representations
    # for 15 these are the same:
    #   +++++++++++++++
    #   <+++[->+++++<]>
    # the second part of the condition was not factually confirmed (v > 240 part)
    if v < 16 or v > 240:
        return value_to_bf_const(v)

    return value_to_bf_mult(v)

def string_to_bf_perchar(text: str, method):
    bf = '>'
    bf += '>'.join(method(c) for c in text)
    return bf

def compress(bf: str):
    "The most basic form of BF program compression"
    last_len, new_len = len(bf), 0
    while new_len < last_len:
        last_len = len(bf)
        bf = bf.replace('<>', '').replace('><', '').replace('-+', '').replace('+-', '')
        new_len = len(bf)
    
    return bf

def get_optimal_factor(value, factor):
    "returns (optimal_factor, remainder)"
    f1 = value // factor
    f2 = f1 + 1
    rem1 = value - factor * f1
    rem2 = value - factor * f2

    return (f1, rem1) if f1 + rem1 < f2 + abs(rem2) else (f2, rem2)

def string_to_bf_clustered(text: list[int]):
    start_time = perf_counter()

    base_conversion = ['>' + value_to_bf_optimal(c) for c in text]
    if verbose: 
        print(f'Calculated base conversion: {len(base_conversion)} characters; total {sum([len(x) for x in base_conversion])} ' +
              f'instructions (avg {sum([len(x) for x in base_conversion]) / len(base_conversion)})')

    # precompute assets
    optimal_factors = [[get_optimal_factor(x, factor) for x in text] for factor in range(2, 128)]
    optimal_factors = [None, None] + [
        (
            x, 
            [sum([f for f, r in x[:i]]) for i in range(len(x) + 1)],
            [sum([abs(r) for f, r in x[:i]]) for i in range(len(x) + 1)],
            [sum(1 for _ in takewhile(lambda fr: fr[1] == 0, x[i:])) for i in range(len(x))]
        ) for x in optimal_factors
    ]

    clusters = []

    # try making clusters for every character except for last (since it has no one to cluster with; clusters only go left->right)
    for index in range(len(text) - 1):
        for last_index in range(index + 1, len(text)):
            cluster_size = last_index - index + 1
            uncompressed_size = sum([len(x) for x in base_conversion[index:last_index + 1]])
            characters = text[index:last_index + 1]
            best_cluster = { 'code': '', 'score': -1 }
            best_score = -1

            # clusters wouldn't make sense for common factor of 1
            for factor in range(2, min(128, ceil(sum(characters) / cluster_size))):
                init_code_size = 1 + cluster_size + factor
                factor_remainders, factor_sums, remainder_sums, tailing_zeros = optimal_factors[factor]
                factors, remainders = zip(*factor_remainders[index:last_index + 1])
                loop_code = 3 + 2 * cluster_size + factor_sums[last_index + 1] - factor_sums[index] # sum(factors)
                leading_zeros = min(tailing_zeros[index], cluster_size - 1) # leading_zeros = sum(1 for _ in takewhile(lambda x: x == 0, remainders[:-1]))
                remainder_code_size = 1 + 2 * (cluster_size - leading_zeros - 1) + remainder_sums[last_index + 1] - remainder_sums[index]

                cluster_code_size = init_code_size + loop_code + remainder_code_size
                cluster_score = uncompressed_size - cluster_code_size

                if cluster_score > best_score:
                    best_cluster = { 'score': cluster_score, 'data': (factor, factors, remainders) }
                    best_score = cluster_score

            best_cluster['avg_score'] = best_cluster['score'] / cluster_size
            best_cluster['size'] = cluster_size
            best_cluster['start_index'] = index

            if best_cluster['score'] > 0:
                if verbose:
                    print(f'Adding a new cluster: [{index};{last_index}] "{"".join(map(chr, text[index:last_index + 1]))}" (length: {cluster_size}), ' +
                          f'score: {best_cluster["score"]} (avg {best_cluster["avg_score"]})')
                clusters.append(best_cluster)
            else:
                if verbose:
                    print(f'Failed to make a good cluster: [{index};{last_index}] (length: {cluster_size}; score: {best_cluster["score"]})')
                # if we encounter a cluster config with no positive score, assume growing this cluster is a bad idea and break early
                if best_cluster['score'] < 0:
                    break

    # now we need to arrange clusters in the best way

    # put clusters in index map: cluster_map[0] gives clusters that start at character 0, etc.
    cluster_map = [[c for c in clusters if c['start_index'] == index] for index in range(len(text))]
    # add base_conversion into cluster_map
    for i, c in enumerate(base_conversion):
        cluster_map[i].append({'code': c, 'size': 1, 'score': 0, 'avg_score': 0, 'start_index': i})

    print(f'Cluster building took {perf_counter() - start_time:0.2f}s')
    start_time = perf_counter()

    def build_best_sequence_dp():
        sequences, scores = [[]], [0]
    
        for target_index in range(len(text)):
            current_score = scores[-1]
            current_sequence = []

            # look for cluster starting at base and ending at new_index that are better than what we have already
            for base in range(0, target_index + 1):
                # there should only be 1 or 0 matching clusters
                matching_cluster = next((c for c in cluster_map[base] if c['size'] == target_index - base + 1), None)
                if matching_cluster is None: continue

                # figure out resulting score with new cluster
                new_score = scores[base] + matching_cluster['score']
                if new_score >= current_score:
                    current_score = new_score
                    current_sequence = [*sequences[base], matching_cluster]

            sequences.append(current_sequence)
            scores.append(current_score)

        return sequences[-1]

    best_sequence = build_best_sequence_dp()

    if verbose:
        print(f'Assembled {len(best_sequence)} clusters in sequence:')
        for cluster in best_sequence:
            index = cluster['start_index']
            last_index = cluster['size'] + index - 1
            print(f'Cluster: [{index};{last_index}] "{"".join(map(chr, text[index:last_index + 1]))}"; ' +
                  f'score: {cluster["score"]} (avg {cluster["avg_score"]})')

    programs = []
    for cluster in best_sequence:
        cluster_size = cluster['size']
        
        if cluster_size == 1:
            programs.append(cluster['code'])
            continue

        factor, factors, remainders = cluster['data']

        init_code = '>' + '>' * cluster_size + '+' * factor
        loop_code = '[-' + ''.join(['<' + '+' * f for f in factors[::-1]]) + '>' * cluster_size + ']'

        leading_zeros = sum(1 for _ in takewhile(lambda x: x == 0, remainders[:-1]))
        remainder_code = ''.join(['<' + signed_value_to_bf(r) for r in remainders[leading_zeros:][::-1]])
        return_code = (len(remainders) - leading_zeros - 1) * '>'
        cluster_code = init_code + loop_code + remainder_code + return_code

        programs.append(cluster_code)

    print(f'Cluster assembly took {1000 * (perf_counter() - start_time):0.2f}ms')

    return ''.join(programs)

def convert_data_to_bf_in_memory(input_data, algo='clustered'):
    program = ''

    if algo == 'clustered':
        program = string_to_bf_clustered(input_data)
    elif algo == 'perchar':
        program = string_to_bf_perchar(input_data, value_to_bf_optimal)
    elif algo == 'basic':
        program = string_to_bf_perchar(input_data, value_to_bf_const)
    else:
        raise Exception(f'Unknown algo: {algo}')

    return program + '[<]>[.>]'

def print_program(program, input_len, name='default'):
    program = compress(program)
    base_size = len(program)

    print(f'Resulting program ({name}): \n\n{program}')
    print(f'\nBase program size: {base_size} instructions (avg: {base_size / input_len:0.2f} inst/char)')

def main():
    global verbose

    input_presets = [
        'Hello world, this is just a test; over...',
        # ~N/A s (new ~50s)
        'put Ten in ten_tmp and overflow_flag; select ten_tmp subtract ten from LB; keep track of overflow compute is_gb_zero into ten_tmp; select overflow_flag add overflow_flag to ten_tmp; keep overflow_flag value on overflow: decrement GB; clear overflow_flag select ten_tmp; ten_tmp = ten_tmp == 2; (stop_condition) set do_div; select ten_tmp on stop_condition: reset do_div; increment LB by 10 copy do_div into overflow_flag; select overflow_flag',
        # ~44.9s (new: ~4.2s)
        'put Ten in ten_tmp and overflow_flag; select ten_tmp subtract ten from LB; keep track of overflow compute is_gb_zero into ten_tmp; select overflow_flag add overflow_flag to ten_tmp;',
        # ~11.0s (new: ~1s)
        'Ok, now, time to throw a real test at this! In particular: a VERY LARGE (not actually but decent) PIECE OF TEXT!',
        # ~1.0s (new ~0.13s)
        'VeRy RaNdOm 0912846510 {are you here, listening???}',
    ]

    parser = argparse.ArgumentParser(description="Convert text to a brainfuck program")

    parser.add_argument('text', nargs='?', default=None, help='Text to convert')
    parser.add_argument('--file', '-f', type=str, help='Take input from a file instead')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--algo', '-a', choices=['clustered', 'perchar', 'basic'], default=[], action='append', help='Conversion algorithm (multiple allowed)')
    parser.add_argument('--mode', '-m', choices=['just-print', 'in-memory'], default='in-memory', help='Program type to generate')
    parser.add_argument('--preset', type=int, help=f'Use a preset string as input. Presets available: {len(input_presets)}')

    args = parser.parse_args()
    verbose = args.verbose
    
    # get input list of values
    input_methods = sum([args.text is not None, args.file is not None, args.preset is not None])
    input_data = None
    if input_methods > 1:
        print('Error: several input methods are specified')
        exit(1)
    elif input_methods == 0:
        input_data = text_to_value_list(input('Input a string you would like to encode in BF: '))
    elif args.text is not None:
        input_data = text_to_value_list(args.text)
    elif args.preset is not None:
        if args.preset >= len(input_presets) or args.preset < 0:
            print(f'Error: invalid preset index {args.preset}')
            exit(1)

        input_data = text_to_value_list(input_presets[args.preset])
        print(f'Using preset string (len: {len(input_data)}): "{input_presets[args.preset]}"')
    elif args.file is not None:
        try:
            with open(args.file, mode='rb') as file:
                input_data = list(file.read())
                print(f'Read {len(input_data)} bytes from a file: {args.file}')
        except e:
            print(f'Error: failed to read file {args.file}: {e}')
            exit(1)

    # generate one or more programs for given input
    programs = []

    if args.mode == 'in-memory':
        algos = args.algo if len(args.algo) > 0 else ['clustered']
        programs = [(convert_data_to_bf_in_memory(input_data, algo), algo) for algo in algos]
    else:
        print(f'Unsupported mode: {args.mode}')
    
    # print the results
    for program, algo in programs:
        print_program(program, len(input_data), algo)

if __name__ == "__main__":
    main()
