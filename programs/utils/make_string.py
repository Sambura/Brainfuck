from math import sqrt, ceil
from itertools import takewhile
from time import perf_counter
import argparse
import numpy as np
import random

from interpreter import bf_execute

verbose_level = 0
perchar_optimal_bf = None
perchar_optimal_bf_lens = None
"Precomputed optimal encodings of every value in [0;255] range"

def text_to_value_list(text: str) -> list[int]:
    "Convert a string to a list of ints (character values)"
    return list(map(ord, text))

def signed_value_to_bf(v: int) -> str:
    "Returns v pluses for positive v and -v minuses for negative v"
    return '+' * v if v >= 0 else '-' * (-v)

def value_to_bf_const(v: int) -> str:
    return signed_value_to_bf(v) if v < 128 else signed_value_to_bf(v - 256)

def signed_value_to_bf_mult(v):
    if v == 0:
        return ''

    av = abs(v)
    f1 = int(sqrt(av))
    f2 = av // f1
    rem = av - f1 * f2
    rem2 = abs(av - f1 * (f2 + 1))
    primary_inst = '+' if v > 0 else '-'
    secondary_inst = '-' if v > 0 else '+'

    if rem <= rem2 + 1:
        return f'>{"+" * f1}[-<{primary_inst * f2}>]<{primary_inst * rem}'

    return f'>{"+" * f1}[-<{primary_inst * (f2 + 1)}>]<{secondary_inst * rem2}'

def value_to_bf_optimal(v: int) -> str:
    # for values below 16 sequence of pluses is the same or often more efficient than multiplication-based representations
    # for 15 these are the same:
    #   +++++++++++++++
    #   <+++[->+++++<]>
    if v < 16 or v > 240:
        return value_to_bf_const(v)

    return signed_value_to_bf_mult(v if v < 128 else v - 256)

def precompute_optimal_encodings():
    global perchar_optimal_bf, perchar_optimal_bf_lens

    if perchar_optimal_bf is not None: return

    perchar_optimal_bf = [value_to_bf_optimal(x) for x in range(256)]
    perchar_optimal_bf_lens = np.array([len(x) for x in perchar_optimal_bf], dtype=np.int32)

def values_to_bf_perchar(values: list[int], method):
    return ''.join(f'>{method(v)}' for v in values)

def values_to_text(values: list[int], escape_controls=True):
    text = ''.join(map(chr, values))
    return text.replace('\n', '\\n').replace('\r', '\\r') if escape_controls else text

def compress(bf: str):
    "The most basic form of BF program compression"
    last_len, new_len = len(bf), 0
    while new_len < last_len:
        last_len = len(bf)
        bf = bf.replace('<>', '').replace('><', '').replace('-+', '').replace('+-', '')
        new_len = len(bf)
    
    return bf

def get_optimal_factor(value: int, factor: int) -> tuple[int, int]:
    "returns (optimal_factor, remainder)"
    f1 = value // factor
    f2 = f1 + 1
    rem1 = value - factor * f1
    rem2 = value - factor * f2

    return (f1, rem1) if (f1 + rem1 < f2 + abs(rem2) or f2 * factor > 255) else (f2, rem2)

def verify_program(program: str, expected_output: list[int], return_details=False) -> bool:
    start_time = perf_counter()
    try:
        output = bf_execute(program, [])
    except:
        print('Encountered error during verification')
        return False

    end_time = perf_counter()
    if verbose > 0:
        print(f'Verification took {1000 * (perf_counter() - start_time):0.1f}ms')

    result = len(expected_output) == len(output) and np.all([x == y for x, y in zip(expected_output, output)])

    return (result, output) if return_details else result

def generate_test_input():
    length = random.randrange(5, 500)

    if random.randrange(0, 2) < 1:
        # just random (except null bytes)
        return (np.random.rand(length) * 254 + 1).astype(int).tolist()

    # these are mostly arbitrary
    ctl = np.random.rand(length) * 54 + 10
    caps = np.random.rand(length) * 32 + 65
    lets = np.random.rand(length) * 29 + 96
    it1 = np.where(np.random.rand(length) < 0.8, lets, caps)
    it2 = np.where(np.random.rand(length) < 0.92, it1, ctl)

    return it2.astype(int).tolist()

def string_to_bf_segmented(text: list[int]):
    fn_start_time = perf_counter()

    precompute_optimal_encodings()
    text_size = len(text)
    base_conversion = ['>' + perchar_optimal_bf[c] for c in text]
    if verbose > 0: 
        print(f'Calculated base conversion: {text_size} characters; total {sum([len(x) for x in base_conversion])} ' +
              f'instructions (avg {np.mean([len(x) for x in base_conversion]):0.2f})')
        start_time = perf_counter()

    # precompute assets
    optimal_factors = np.array([[get_optimal_factor(x, factor) for x in text] for factor in range(2, 128)]) # (126, text_size, 2)
    factor_range = optimal_factors.shape[0]
    factor_sums = np.zeros((factor_range, text_size + 1), dtype=np.int64)
    factor_sums[:, 1:] = np.cumsum(optimal_factors[:, :, 0], axis=1)
    remainder_sums = np.zeros((factor_range, text_size + 1), dtype=np.int64)
    remainder_sums[:, 1:] = np.cumsum(np.abs(optimal_factors[:, :, 1]), axis=1)
    tailing_zeros = np.array([[sum(1 for _ in takewhile(lambda r: r == 0, x[i:, 1])) for i in range(len(x))] for x in optimal_factors])
    if text_size > 4:
        precomputed_escapes = np.zeros_like(tailing_zeros)
        precomputed_escape_starts = np.zeros_like(tailing_zeros)
        escape_ramp = np.arange(1, 4) - 5
    
        for factor_index in range(128 - 2):
            l_factors = optimal_factors[factor_index, :, 0].tolist()
            pes = precomputed_escapes[factor_index]
            pess = precomputed_escape_starts[factor_index]
            escape_size = 0
            since_last_zero = 0
            glider_reset = False

            for i, factor in enumerate(l_factors):
                pes[i] = escape_size
                pess[i] = escape_size - 1

                if factor == 0:
                    escape_size += 1
                    since_last_zero = 0
                elif since_last_zero < 3:
                    escape_size += 1
                    since_last_zero += 1
                elif since_last_zero < 4:
                    since_last_zero = 4

                # glider=T: escape_size - 3
                # glider=F,F,F,T: escape_size - 3 for 3? elements prior first glider
                # glider=T,F,F,F: escape_size + escape_ramp ending at last glider
                # everywhere: -1, to account for glider reset/counter shift at the end of the range

                if since_last_zero == 4:
                    since_last_zero += 1
                    pess[i - 3: i + 1] = escape_size - 4
                    glider_reset = True
                elif since_last_zero > 4:
                    pess[i] = escape_size - 4
                elif glider_reset: # current index is the last glider
                    glider_reset = False
                    pess[i - 2:i + 1] = escape_size + escape_ramp

    def get_glide_code_size(distance):
        overflows = distance // 255
        rest = distance - overflows * 255
        overflow_size = 8 + 3 * overflows if overflows > 0 else 0
        return overflow_size + min(rest, 10 + perchar_optimal_bf_lens[rest])

    uncompressed_size_sums = [0] + np.cumsum([len(x) for x in base_conversion]).tolist()
    character_sums = [0] + np.cumsum(text).tolist()
    glide_code_sizes = list(map(get_glide_code_size, range(text_size + 1)))

    if verbose > 0:
        print(f'Preprocessing took {perf_counter() - start_time:0.2f}s')
        start_time = perf_counter()

    # segment_map[start_index][end_index] = segment
    segment_map = [[None] * text_size for _ in range(text_size)]
    for i, c in enumerate(base_conversion):
        segment_map[i][i] = {'code': c, 'score': 0, 'start_index': i, 'size': 1}
    segment_count = text_size

    def avg_score(segment):
        return segment['score'] / segment['size']

    # try making segments for every character except for last (since it has no one to group with; segments only go left->right)
    for index in range(text_size - 1):
        for last_index in range(index + 1, text_size):
            segment_size = last_index - index + 1
            uncompressed_size = uncompressed_size_sums[last_index + 1] - uncompressed_size_sums[index]

            # segments wouldn't make sense for common factor of 1
            max_factor = max(3, min(128, ceil((character_sums[last_index + 1] - character_sums[index]) / segment_size)))
            init_code_sizes = 1 + glide_code_sizes[segment_size] + perchar_optimal_bf_lens[2:max_factor]
            factor_sums_l = factor_sums[:max_factor - 2]
            remainder_sums_l = remainder_sums[:max_factor - 2]
            tailing_zeros_l = tailing_zeros[:max_factor - 2]

            if segment_size > 4:
                escape_code_sizes = precomputed_escapes[:max_factor - 2, last_index] - precomputed_escape_starts[:max_factor - 2, index]
            else:
                escape_code_sizes = np.ones_like(init_code_sizes) * segment_size

            loop_code_sizes = 3 + segment_size + escape_code_sizes + factor_sums_l[:, last_index + 1] - factor_sums_l[:, index]
            leading_zeros = np.minimum(tailing_zeros_l[:, index], segment_size - 1)
            return_shifts = np.minimum(segment_size - 1 - leading_zeros, 4)
            remainder_code_sizes = segment_size - leading_zeros + return_shifts + remainder_sums_l[:, last_index + 1] - remainder_sums_l[:, index]
            segment_code_sizes = init_code_sizes + loop_code_sizes + remainder_code_sizes
            segment_scores = uncompressed_size - segment_code_sizes
            best_index = np.argmax(segment_scores)
            segment_score = segment_scores[best_index]

            best_segment = {
                'score': segment_score, 
                'factor': best_index + 2,
                'start_index': index,
                'size': segment_size,
                # for debug purposes
                # 'code_size': segment_code_sizes[best_index],
                # 'breakdown': {
                #     'init': init_code_sizes[best_index],
                #     'escape': escape_code_sizes[best_index],
                #     'loop': loop_code_sizes[best_index] - escape_code_sizes[best_index],
                #     'return': return_shifts[best_index],
                #     'rems': remainder_code_sizes[best_index] - return_shifts[best_index],
                # }
            }

            if segment_score > 0:
                if verbose > 3:
                    print(f'Adding a new segment (best_index: {best_index}): [{index};{last_index}] "{values_to_text(text[index:last_index + 1])}" ' +
                          f'(length: {segment_size}), score: {best_segment["score"]} (avg {avg_score(best_segment)})')
                segment_map[index][last_index] = best_segment
                segment_count += 1
            else:
                if verbose > 3:
                    print(f'Failed to make a good segment: [{index};{last_index}] (length: {segment_size}; score: {best_segment["score"]})')
                # if we encounter a segment config with no positive score, assume growing this segment is a bad idea and break early
                if best_segment['score'] < 0:
                    break

    if verbose > 0:
        print(f'Built {segment_count} segments. Segment building took {perf_counter() - start_time:0.2f}s')
        start_time = perf_counter()

    # now we need to arrange segments in the best way

    def build_best_sequence_dp():
        sequences, scores = [[]], [0]
    
        for target_index in range(text_size):
            current_score = -1
            current_sequence = None

            # look for segment starting at base and ending at target_index that are better than what we have already
            for base in range(target_index + 1):
                matching_segment = segment_map[base][target_index]
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
        print(f'Assembled {len(best_sequence)} segments in sequence (total score: {sum([x["score"] for x in best_sequence])})')

        if verbose > 1:
            for segment in best_sequence:
                index, size = segment['start_index'], segment['size']
                last_index = size + index - 1
                print(f'Segment: [{index};{last_index}] "{values_to_text(text[index:last_index + 1])}"; ' +
                      f'score: {segment["score"]} (avg {avg_score(segment):0.2f})')

        print(f'Segment assembly took {perf_counter() - start_time:0.2f}s')
        start_time = perf_counter()

    program_segments = []
    for segment in best_sequence:
        start_index, segment_size = segment['start_index'], segment['size']
        
        if segment_size == 1:
            program_segments.append(segment['code'])
            continue

        factor = segment['factor']
        factor_index, end_index = factor - 2, start_index + segment_size
        factors: list[int] = optimal_factors[factor_index, start_index:end_index, 0].tolist()
        remainders: list[int] = optimal_factors[factor_index, start_index:end_index, 1].tolist()

        travel_distance = segment_size
        engage_code = '>'

        while travel_distance > 0:
            if travel_distance >= 255:
                overflow_count = travel_distance // 255
                engage_code += f'-[[-{">" * overflow_count}+{"<" * overflow_count}]{">" * overflow_count}-]'
                travel_distance -= 255 * overflow_count
            # 30 is the first number that has optimal representation that is 10+ instructions more efficient than basic;
            # since glider has overhead of 10 instructions, we need to have at least 10 spare to use it effectively
            elif travel_distance >= 30:
                engage_code += perchar_optimal_bf[travel_distance] + '[[->+<]>-]'
                break
            else:
                engage_code += '>' * travel_distance
                break

        init_code = engage_code + perchar_optimal_bf[factor]

        # make code to end loop, should be equivalent to '>' * segment_size
        # can be replaced with [>]< unless segment_size is below 4 or there are zeros in `factors`
        loop_escape_code = ''
        current_shift = 0
        while current_shift < segment_size:
            if 0 not in factors[current_shift:]:
                remaining_distance = segment_size - current_shift
                loop_escape_code += '[>]<' if remaining_distance > 4 else '>' * remaining_distance
                break

            first_zero = factors.index(0, current_shift)
            zero_distance = first_zero - current_shift
            loop_escape_code += '[>]' if zero_distance > 3 else '>' * zero_distance
            loop_escape_code += '>'
            current_shift = first_zero + 1

        loop_code = '[' + ''.join(['<' + '+' * f for f in factors[::-1]]) + loop_escape_code + '-]'

        leading_zeros = sum(1 for _ in takewhile(lambda x: x == 0, remainders[:-1]))
        remainder_code = ''.join(['<' + signed_value_to_bf(r) for r in remainders[leading_zeros:][::-1]])
        return_shifts = len(remainders) - leading_zeros - 1
        return_code = return_shifts * '>' if return_shifts <= 4 else '[>]<'
        segment_code = init_code + loop_code + remainder_code + return_code

        # if we save code_size, ensure it matches with actual generated program
        if 'code_size' in segment and len(segment_code) != segment['code_size']:
            print(f'Segment size mismatch: [{start_index}:{start_index + segment_size - 1}] precomputed {segment["code_size"]} vs. actual {len(segment_code)}')
            print(f'Precomputed breakdown: ' + ', '.join([f'{key}: {size}' for key, size in segment['breakdown'].items()]))
            print(f'Generated segment code: {segment_code}')

            raise Exception('Segment code mismatch')

        program_segments.append(segment_code)

    program = ''.join(program_segments)

    if verbose > 0:
        print(f'Code generation took {perf_counter() - start_time:0.2f}s. Total time elapsed: {perf_counter() - fn_start_time:0.2f}s')

    return program

def run_self_test(program_generator, args, run_count=250) -> bool:
    print(f'Executing self test: {run_count} runs with args: {args}')

    start_time = perf_counter()
    exception = None
    for _ in range(run_count):
        test_input = generate_test_input()
        try:
            test_program = program_generator(test_input, *args)
            ok, test_output = verify_program(test_program, test_input, return_details=True)
        except Exception as e:
            ok = False
            test_output = []
            test_program = ''
            exception = e
            print(f'\nHalting self test due to exception: {e}\n')

        if not ok:
            print(f'Self test failed. Test input was: ({len(test_input)}) {test_input}')
            print(f'Or:\n\n{values_to_text(test_input)}\n')
            print(f'Test output was: {test_output}')
            print(f'Or:\n\n{values_to_text(test_output)}\n')
            print(f'Program generated for input: {test_program}')

            if exception is not None:
                print('Re-throwing the exception:')
                raise exception

            return False

    print(f'Self test OK. Time elapsed {perf_counter() - start_time:0.1f}s')

    return True

def convert_data_to_bf_in_memory(input_data, algo='segmented'):
    program = ''

    if algo == 'segmented':
        program = string_to_bf_segmented(input_data)
    elif algo == 'perchar':
        program = values_to_bf_perchar(input_data, value_to_bf_optimal)
    elif algo == 'basic':
        program = values_to_bf_perchar(input_data, value_to_bf_const)
    else:
        raise Exception(f'Unknown algo: {algo}')

    return program + '[<]>[.>]'

def print_program(program, input_len, no_program, name='default'):
    program = compress(program)
    base_size = len(program)

    if not no_program:
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
    parser.add_argument('--no-program', action='store_true', help='Do not output resulting program, only debug messages')
    parser.add_argument('--no-verify', action='store_true', help='Do not verify the generated program for correctness')
    parser.add_argument('--run-test', nargs='?', const=250, default=None, type=int, help='Test algorithm on random inputs, optionally specify number of runs')

    args = parser.parse_args()
    verbose = args.verbose
    
    # get input list of values
    input_methods = sum([args.text is not None, args.file is not None, args.preset is not None, args.run_test is not None])
    input_data = []
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

    if 0 in input_data:
        print(f'Error: Null bytes are not allowed in input data (at index {input_data.index(0)})')
        exit(1)

    # generate one or more programs for given input
    generator = None

    if args.mode == 'in-memory':
        algos = args.algo if len(args.algo) > 0 else ['segmented']
        generator = convert_data_to_bf_in_memory
    else:
        print(f'Unsupported mode: {args.mode}')
        exit(1)

    if args.run_test is not None:
        for algo in algos:
            if not run_self_test(generator, (algo, ), run_count=args.run_test):
                exit(1)
        exit(0)

    programs = [(generator(input_data, algo), algo) for algo in algos]

    # verify and print the results
    for program, algo in programs:
        if not args.no_verify:
            ok = verify_program(program, input_data)
            if not ok:
                print(f'Error: verification failed for program generated by algo {algo}')
                exit(1)

        print_program(program, len(input_data), args.no_program, algo)

if __name__ == "__main__":
    main()
