from math import sqrt, ceil
from itertools import takewhile
from time import perf_counter
import argparse
import numpy as np

verbose_level = 0

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

def string_to_bf_segmented(text: list[int]):
    fn_start_time = perf_counter()

    base_conversion = ['>' + value_to_bf_optimal(c) for c in text]
    if verbose > 0: 
        print(f'Calculated base conversion: {len(base_conversion)} characters; total {sum([len(x) for x in base_conversion])} ' +
              f'instructions (avg {sum([len(x) for x in base_conversion]) / len(base_conversion):0.2f})')
        start_time = perf_counter()

    # precompute assets
    optimal_factors = [[get_optimal_factor(x, factor) for x in text] for factor in range(2, 128)]
    factor_sums = np.array([[sum([f for f, r in x[:i]]) for i in range(len(x) + 1)] for x in optimal_factors])
    remainder_sums = np.array([[sum([abs(r) for f, r in x[:i]]) for i in range(len(x) + 1)] for x in optimal_factors])
    tailing_zeros = np.array([[sum(1 for _ in takewhile(lambda fr: fr[1] == 0, x[i:])) for i in range(len(x))] for x in optimal_factors])
    optimal_factors = np.array([list(zip(*x)) for x in optimal_factors], dtype=np.int16)

    uncompressed_size_sums = np.zeros(len(text) + 1, dtype=np.int64)
    uncompressed_size_sums[1:] = np.cumsum([len(x) for x in base_conversion])
    character_sums = np.zeros(len(text) + 1, dtype=np.int64)
    character_sums[1:] = np.cumsum(text)

    if verbose > 0:
        print(f'Preprocessing took {perf_counter() - start_time:0.2f}s')
        start_time = perf_counter()

    segments = []

    # try making segments for every character except for last (since it has no one to group with; segments only go left->right)
    for index in range(len(text) - 1):
        for last_index in range(index + 1, len(text)):
            segment_size = last_index - index + 1
            uncompressed_size = uncompressed_size_sums[last_index + 1] - uncompressed_size_sums[index]

            # segments wouldn't make sense for common factor of 1
            max_factor = min(128, ceil((character_sums[last_index + 1] - character_sums[index]) / segment_size))
            p_factors = np.arange(2, max_factor, dtype=np.int32)
            init_code_sizes = 1 + segment_size + p_factors
            factor_sums_l = factor_sums[:max_factor - 2]
            remainder_sums_l = remainder_sums[:max_factor - 2]
            tailing_zeros_l = tailing_zeros[:max_factor - 2]
            loop_code_sizes = 3 + 2 * segment_size + factor_sums_l[:, last_index + 1] - factor_sums_l[:, index]
            leading_zeros = np.minimum(tailing_zeros_l[:, index], segment_size - 1)
            remainder_code_sizes = 1 + 2 * (segment_size - leading_zeros - 1) + remainder_sums_l[:, last_index + 1] - remainder_sums_l[:, index]
            segment_code_sizes = init_code_sizes + loop_code_sizes + remainder_code_sizes
            segment_scores = uncompressed_size - segment_code_sizes
            best_index = np.argmax(segment_scores)
            segment_score = segment_scores[best_index]

            best_segment = {
                'score': segment_score, 
                'data': (p_factors[best_index], optimal_factors[best_index, 0, index:last_index + 1], optimal_factors[best_index, 1, index:last_index + 1]),
                'avg_score': segment_score / segment_size,
                'size': segment_size,
                'start_index': index
            }

            if segment_score > 0:
                if verbose > 3:
                    print(f'Adding a new segment: [{index};{last_index}] "{"".join(map(chr, text[index:last_index + 1]))}" (length: {segment_size}), ' +
                          f'score: {best_segment["score"]} (avg {best_segment["avg_score"]})')
                segments.append(best_segment)
            else:
                if verbose > 3:
                    print(f'Failed to make a good segment: [{index};{last_index}] (length: {segment_size}; score: {best_segment["score"]})')
                # if we encounter a segment config with no positive score, assume growing this segment is a bad idea and break early
                if best_segment['score'] < 0:
                    break

    # now we need to arrange segments in the best way

    # put segments in index map: segment_map[0] gives segments that start at character 0, etc.
    segment_map = [[s for s in segments if s['start_index'] == index] for index in range(len(text))]
    # add base_conversion into segment_map
    for i, c in enumerate(base_conversion):
        segment_map[i].append({'code': c, 'size': 1, 'score': 0, 'avg_score': 0, 'start_index': i})

    if verbose > 0:
        print(f'Segment building took {perf_counter() - start_time:0.2f}s')
        print(f'Built {np.sum([len(x) for x in segment_map])} segments')
        start_time = perf_counter()

    def build_best_sequence_dp():
        sequences, scores = [[]], [0]
    
        for target_index in range(len(text)):
            current_score = scores[-1]
            current_sequence = []

            # look for segment starting at base and ending at new_index that are better than what we have already
            for base in range(0, target_index + 1):
                # there should only be 1 or 0 matching segments
                matching_segment = next((s for s in segment_map[base] if s['size'] == target_index - base + 1), None)
                if matching_segment is None: continue

                # figure out resulting score with new segment
                new_score = scores[base] + matching_segment['score']
                if new_score >= current_score:
                    current_score = new_score
                    current_sequence = [*sequences[base], matching_segment]

            sequences.append(current_sequence)
            scores.append(current_score)

        return sequences[-1]

    best_sequence = build_best_sequence_dp()

    if verbose > 0:
        print(f'Assembled {len(best_sequence)} segments in sequence')

        if verbose > 1:
            for segment in best_sequence:
                index = segment['start_index']
                last_index = segment['size'] + index - 1
                print(f'Segment: [{index};{last_index}] "{"".join(map(chr, text[index:last_index + 1]))}"; ' +
                      f'score: {segment["score"]} (avg {segment["avg_score"]:0.2f})')

        print(f'Segment assembly took {perf_counter() - start_time:0.2f}s')
        start_time = perf_counter()

    program_segments = []
    for segment in best_sequence:
        segment_size = segment['size']
        
        if segment_size == 1:
            program_segments.append(segment['code'])
            continue

        factor, factors, remainders = segment['data']

        init_code = '>' + '>' * segment_size + '+' * factor
        loop_code = '[-' + ''.join(['<' + '+' * f for f in factors[::-1]]) + '>' * segment_size + ']'

        leading_zeros = sum(1 for _ in takewhile(lambda x: x == 0, remainders[:-1]))
        remainder_code = ''.join(['<' + signed_value_to_bf(r) for r in remainders[leading_zeros:][::-1]])
        return_code = (len(remainders) - leading_zeros - 1) * '>'
        segment_code = init_code + loop_code + remainder_code + return_code

        program_segments.append(segment_code)

    program = ''.join(program_segments)

    if verbose > 0:
        print(f'Code generation took {perf_counter() - start_time:0.2f}s. Total time elapsed: {perf_counter() - fn_start_time:0.2f}s')

    return program

def convert_data_to_bf_in_memory(input_data, algo='segmented'):
    program = ''

    if algo == 'segmented':
        program = string_to_bf_segmented(input_data)
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
        # ~N/A s (new ~50s) (np v5: ~2.5s)
        'put Ten in ten_tmp and overflow_flag; select ten_tmp subtract ten from LB; keep track of overflow compute is_gb_zero into ten_tmp; select overflow_flag add overflow_flag to ten_tmp; keep overflow_flag value on overflow: decrement GB; clear overflow_flag select ten_tmp; ten_tmp = ten_tmp == 2; (stop_condition) set do_div; select ten_tmp on stop_condition: reset do_div; increment LB by 10 copy do_div into overflow_flag; select overflow_flag',
        # ~44.9s (new: ~4.2s) (np v5: ~0.25s)
        'put Ten in ten_tmp and overflow_flag; select ten_tmp subtract ten from LB; keep track of overflow compute is_gb_zero into ten_tmp; select overflow_flag add overflow_flag to ten_tmp;',
        # ~11.0s (new: ~1s) (np v5: ~0.1s)
        'Ok, now, time to throw a real test at this! In particular: a VERY LARGE (not actually but decent) PIECE OF TEXT!',
        # ~1.0s (new ~0.13s) (np v5: ~0.02s)
        'VeRy RaNdOm 0912846510 {are you here, listening???}',
    ]

    parser = argparse.ArgumentParser(description="Convert text to a brainfuck program")

    parser.add_argument('text', nargs='?', default=None, help='Text to convert')
    parser.add_argument('--file', '-f', type=str, help='Take input from a file instead')
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Enable verbose output. Repeat to increase verbosity level')
    parser.add_argument('--algo', '-a', choices=['segmented', 'perchar', 'basic'], default=[], action='append', help='Conversion algorithm (multiple allowed)')
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
        except Exception as e:
            print(f'Error: failed to read file {args.file}: {e}')
            exit(1)

    # generate one or more programs for given input
    programs = []

    if args.mode == 'in-memory':
        algos = args.algo if len(args.algo) > 0 else ['segmented']
        programs = [(convert_data_to_bf_in_memory(input_data, algo), algo) for algo in algos]
    else:
        print(f'Unsupported mode: {args.mode}')
    
    # print the results
    for program, algo in programs:
        print_program(program, len(input_data), algo)

if __name__ == "__main__":
    main()
