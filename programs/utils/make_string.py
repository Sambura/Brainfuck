from math import sqrt, ceil
from itertools import takewhile
from time import perf_counter
import argparse
import numpy as np
import random

from interpreter import bf_execute

verbose = 0
perchar_optimal_bf = None
perchar_optimal_bf_lens = None
"Precomputed optimal encodings of every value in [0;255] range"
perchar_optimal_bf_drift = None
perchar_optimal_bf_drift_lens = None

def text_to_value_list(text: str) -> list[int]:
    "Convert a string to a list of ints (character values)"
    return list(map(ord, text))

def signed_value_to_bf(v: int) -> str:
    "Returns v pluses for positive v and -v minuses for negative v"
    return '+' * v if v >= 0 else '-' * (-v)

def value_to_bf_const(v: int) -> str:
    return signed_value_to_bf(v) if v < 128 else signed_value_to_bf(v - 256)

def signed_value_to_bf_mult(v: int, no_drift: bool=True) -> str:
    """ Returns brainfuck code to store value v to memory. Requires one empty cell to the right of the current one.
        The value is stored to the `new current cell`. `new current cell` is the current cell if no_drift is True,
        and is the cell to the right of the current one if no_drift is False. no_drift == True increases code size
        by 1 instruction                                                                                            """
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
        if no_drift:
            return f'>{"+" * f1}[-<{primary_inst * f2}>]<{primary_inst * rem}'
        else:
            return f'{"+" * f1}[->{primary_inst * f2}<]>{primary_inst * rem}'

    if no_drift:
        return f'>{"+" * f1}[-<{primary_inst * (f2 + 1)}>]<{secondary_inst * rem2}'
    else:
        return f'{"+" * f1}[->{primary_inst * (f2 + 1)}<]>{secondary_inst * rem2}'

def value_to_bf_optimal(v: int, no_drift: bool=True) -> str:
    const = value_to_bf_const(v)
    mult = signed_value_to_bf_mult(v if v < 128 else v - 256, no_drift=no_drift)
    return const if len(const) <= len(mult) else mult

def precompute_optimal_encodings():
    global perchar_optimal_bf, perchar_optimal_bf_lens, perchar_optimal_bf_drift, perchar_optimal_bf_drift_lens

    if perchar_optimal_bf is not None: return

    perchar_optimal_bf = [value_to_bf_optimal(x) for x in range(256)]
    perchar_optimal_bf_lens = np.array([len(x) for x in perchar_optimal_bf], dtype=np.int32)

    perchar_optimal_bf_drift = [value_to_bf_optimal(x, no_drift=False) for x in range(256)]
    perchar_optimal_bf_drift_lens = np.array([len(x) for x in perchar_optimal_bf_drift], dtype=np.int32)

def values_to_bf_perchar(values: list[int], method):
    return ''.join(f'>{method(v)}' for v in values)

def values_to_text(values: list[int], escape_controls=True):
    text = ''.join(map(chr, values))
    return text.replace('\n', '\\n').replace('\r', '\\r') if escape_controls else text

def bf_collapse(program: str) -> str:
    "Collapse any existing `<>` / `><` / `+-` / `-+` "
    last_len, new_len = len(program), 0
    while new_len < last_len:
        last_len = len(program)
        program = program.replace('<>', '').replace('><', '').replace('-+', '').replace('+-', '')
        new_len = len(program)

    return program

def bf_active_shift_collapse(program: str) -> str:
    "Basic brainfuck compression method: reorder and collapse any -+ or <> pairs"
    # TODO: test how good this is and make it actually useful next time
    global_chunks = []
    cur_index = 0

    def find_instructions_index(program: str, instructions: str, index: int=0):
        indices = [program.index(c, index) if c in program[index:] else len(program) for c in instructions]
        return min(*indices)

    def detect_movable_chunk(chunk: str, index: int):
        shift = 1 if chunk[index] == '>' else (-1 if chunk[index] == '<' else 0)

        for i in range(index + 1, len(chunk)):
            if chunk[i] == '>':
                shift += 1
            elif chunk[i] == '<':
                shift -= 1
            elif chunk[i] == '.' or chunk[i] == ',':
                return None

            if shift == 0:
                return i

        return None

    # separate program into chunks with respect to loop boundaries (for now assume we can't reorder loops / beyond loops)
    while cur_index < len(program):
        loop_index = find_instructions_index(program, '[]', cur_index)

        if loop_index >= len(program):
            global_chunks.append(program[cur_index:])
            break

        global_chunks.append(program[cur_index:loop_index])
        global_chunks.append(program[loop_index])
        cur_index = loop_index + 1

    # optimize each chunk separately
    def collapse_chunk(chunk: str) -> str:
        chunk = bf_collapse(chunk)
        if len(chunk) <= 2: # pretty sure we can return anything up to 4 instructions long, but i just can't prove it...
            return chunk    # >+<-> is the shortest chunk i can think of (collapses to ->+)

        scan_start = 0
        while scan_start < len(chunk):
            movable_chunk = None
            start = find_instructions_index(chunk, '<>', scan_start)

            for i in range(start, len(chunk)):
                if chunk[i] == '>' or chunk[i] == '<':
                    chunk_end = detect_movable_chunk(chunk, i)
                    if chunk_end is not None:
                        movable_chunk = (i, chunk_end)
                        break

                elif chunk[i] == '.' or chunk[i] == ',':
                    scan_start = i + 1

            if movable_chunk is None:
                break

            i_start = chunk[movable_chunk[0]]
            i_end = chunk[movable_chunk[1]]
            sub_chunk = chunk[movable_chunk[0]:movable_chunk[1] + 1]

            next_shift_index = find_instructions_index(chunk, '<>', movable_chunk[1] + 1)
            if next_shift_index >= len(chunk):
                next_stop_index = find_instructions_index(chunk, ',.', movable_chunk[1] + 1)
                next_stop_index = min(next_stop_index, len(chunk) - 1)

                if next_stop_index == movable_chunk[1]:
                    scan_start += 1
                    continue

                chunk = chunk[:movable_chunk[0]] + chunk[movable_chunk[1] + 1:next_stop_index + 1] + sub_chunk + chunk[next_stop_index + 1:]

                continue

            if i_start == chunk[next_shift_index]:
                chunk = chunk[:movable_chunk[0]] + chunk[movable_chunk[1] + 1:next_shift_index] + sub_chunk + chunk[next_shift_index:]
            else:
                scan_start += 1
                continue

            chunk = bf_collapse(chunk)

        return chunk

    return ''.join([collapse_chunk(x) for x in global_chunks])

def bf_compress(program: str) -> str:
    "The best we can do right now in terms of compression"

    c_program = bf_active_shift_collapse(program)

    return c_program

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
        return (False, '') if return_details else False

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

def string_to_bf_segmented(text: list[int], no_drift=True, silent=False):
    fn_start_time = perf_counter()
    verbose_local = 0 if silent else verbose

    precompute_optimal_encodings()
    text_size = len(text)
    base_conversion = ['>' + perchar_optimal_bf[c] for c in text]
    if verbose_local > 0: 
        print(f'Calculated base conversion: {text_size} characters; total {sum([len(x) for x in base_conversion])} ' +
              f'instructions (avg {np.mean([len(x) for x in base_conversion]):0.2f})')
        start_time = perf_counter()

    # precompute assets
    common_factor_range_start = 2 # do not modify (yet)
    common_factor_range_end = 128
    optimal_factors = np.array([[get_optimal_factor(x, factor) for x in text] for factor in range(common_factor_range_start, common_factor_range_end)], dtype=np.int_)
    factor_range = optimal_factors.shape[0]
    factor_sums = np.zeros((factor_range, text_size + 1), dtype=np.int_)
    factor_sums[:, 1:] = np.cumsum(optimal_factors[:, :, 0], axis=1)
    remainder_sums = np.zeros((factor_range, text_size + 1), dtype=np.int_)
    remainder_sums[:, 1:] = np.cumsum(np.abs(optimal_factors[:, :, 1]), axis=1)
    tailing_zeros = np.array([[sum(1 for _ in takewhile(lambda r: r == 0, x[i:, 1])) for i in range(len(x))] for x in optimal_factors], dtype=np.int_)
    if text_size > 4:
        precomputed_escapes = np.zeros_like(tailing_zeros)
        precomputed_escape_starts = np.zeros_like(tailing_zeros)
        escape_ramp = np.arange(1, 4) - 5

        for factor_index in range(common_factor_range_end - common_factor_range_start):
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

    def generate_code(segment_sequence):
        program_segments = []
        for segment in segment_sequence:
            start_index, segment_size = segment['start_index'], segment['size']

            if segment_size == 1:
                program_segments.append(segment['code'])
                continue

            factor = segment['factor']
            factor_index, end_index = factor - 2, start_index + segment_size
            factors: list[int] = optimal_factors[factor_index, start_index:end_index, 0].tolist()
            remainders: list[int] = optimal_factors[factor_index, start_index:end_index, 1].tolist()

            if no_drift or len(segment_sequence) > 1:
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
                loop_right = '>'
                loop_left = '<'
                leading_zeros = sum(1 for _ in takewhile(lambda x: x == 0, remainders[:-1]))
            else:
                factors = factors[::-1]
                remainders = remainders[::-1]
                init_code = perchar_optimal_bf_drift[factor]
                if '>' not in init_code: # force 1 cell drift
                    init_code = '>' + init_code
                loop_right = '<'
                loop_left = '>'
                leading_zeros = 0

            # make code to end loop, should be equivalent to '>' * segment_size
            # can be replaced with [>]< unless segment_size is below 4 or there are zeros in `factors`
            loop_escape_code = ''
            current_shift = 0
            while current_shift < segment_size:
                if 0 not in factors[current_shift:]:
                    remaining_distance = segment_size - current_shift
                    loop_escape_code += f'[{loop_right}]{loop_left}' if remaining_distance > 4 else loop_right * remaining_distance
                    break

                first_zero = factors.index(0, current_shift)
                zero_distance = first_zero - current_shift
                loop_escape_code += f'[{loop_right}]' if zero_distance > 3 else loop_right * zero_distance
                loop_escape_code += loop_right
                current_shift = first_zero + 1

            loop_code = '[' + ''.join([loop_left + '+' * f for f in factors[::-1]]) + loop_escape_code + '-]'

            remainder_code = ''.join([loop_left + signed_value_to_bf(r) for r in remainders[leading_zeros:][::-1]])
            return_shifts = len(remainders) - leading_zeros - 1
            return_code = return_shifts * loop_right if return_shifts <= 4 else f'[{loop_right}]{loop_left}'
            if loop_right == '<':
                return_code = ''
            segment_code = init_code + loop_code + remainder_code + return_code

            # if we save code_size, ensure it matches with actual generated program
            if 'code_size' in segment and len(segment_code) != segment['code_size']:
                print(f'Segment size mismatch: [{start_index}:{start_index + segment_size - 1}] precomputed {segment["code_size"]} vs. actual {len(segment_code)}')
                print(f'Precomputed breakdown: ' + ', '.join([f'{key}: {size}' for key, size in segment['breakdown'].items()]))
                print(f'Generated segment code: {segment_code}')

                raise Exception('Segment code mismatch')

            program_segments.append(segment_code)

        return ''.join(program_segments)

    if verbose_local > 0:
        print(f'Preprocessing took {perf_counter() - start_time:0.2f}s')
        start_time = perf_counter()

    def avg_score(segment):
        return segment['score'] / segment['size']

    sequences, seq_scores = [[]], [0]
    mono_segment = None

    for last_index in range(text_size):
        current_sequence = sequences[-1] + [{'code': base_conversion[last_index], 'score': 0, 'start_index': last_index, 'size': 1}]
        current_seq_score = seq_scores[-1]

        for index in range(last_index):
            segment_size = last_index - index + 1
            uncompressed_size = uncompressed_size_sums[last_index + 1] - uncompressed_size_sums[index]

            max_factor = max(3, min(common_factor_range_end, ceil((character_sums[last_index + 1] - character_sums[index]) / segment_size)))
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

            if segment_size == text_size:
                mono_segment = best_segment

            if segment_score > 0:
                if verbose_local > 3:
                    print(f'Adding a new segment (best_index: {best_index}): [{index};{last_index}] "{values_to_text(text[index:last_index + 1])}" ' +
                          f'(length: {segment_size}), score: {best_segment["score"]} (avg {avg_score(best_segment)})')

                new_score = seq_scores[index] + segment_score
                if new_score >= current_seq_score:
                    current_seq_score = new_score
                    current_sequence = [*sequences[index], best_segment]
            else:
                if verbose_local > 3:
                    print(f'Failed to make a good segment: [{index};{last_index}] (length: {segment_size}; score: {best_segment["score"]})')
                # if we encounter a segment config with no positive score, assume growing this segment is a bad idea and break early
                if best_segment['score'] < 0:
                    break

        sequences.append(current_sequence)
        seq_scores.append(current_seq_score)

    if not no_drift and mono_segment is not None:
        mono_segment.pop('code_size', None)
        mono_segment.pop('breakdown', None)
        mono_segment_code = generate_code([mono_segment])
        mono_segment_score = uncompressed_size_sums[-1] - len(mono_segment_code)
        if verbose_local > 0:
            print(f'Built mono-segment: {len(mono_segment_code)} instructions (score: {mono_segment_score})')

        if mono_segment_score > seq_scores[-1]:
            sequences[-1] = [mono_segment]
            seq_scores[-1] = mono_segment_score

    best_sequence: list[dict] = sequences[-1]

    if verbose_local > 0:
        print(f'Built and assembled {len(best_sequence)} segments in sequence. (total score: {seq_scores[-1]}). ' +
              f'Segment building and assembly took {perf_counter() - start_time:0.2f}s')

        if verbose_local > 1:
            for segment in best_sequence:
                index, size = segment['start_index'], segment['size']
                last_index = size + index - 1
                slice_limit = last_index + 1 if verbose_local > 2 else min(last_index + 1, index + 10)
                postfix = '...' if slice_limit < last_index + 1 else ''
                factor = segment.get('factor', -1)
                print(f'Segment: [{index};{last_index}] ({size}) (factor: {factor}) "{values_to_text(text[index:slice_limit])}{postfix}"; ' +
                      f'score: {segment["score"]} (avg {avg_score(segment):0.2f})')

        start_time = perf_counter()

    program = generate_code(best_sequence)

    if verbose_local > 0:
        print(f'Code generation took {perf_counter() - start_time:0.2f}s. Total time elapsed: {perf_counter() - fn_start_time:0.2f}s')

    return program

def bf_print_values_const(text: list[int]) -> str:
    deltas = np.array(text) - np.array([0] + text[:-1])
    return ''.join([f'{signed_value_to_bf(delta)}.' for delta in deltas])

def bf_print_values_ranged(text: list[int], max_memory_cells=257, max_optimal_ranges=7, avoid_loops=False, silent=False) -> str:
    fn_start_time = perf_counter()
    verbose_local = 0 if silent else verbose
    if max_memory_cells < 2:
        raise Exception('max_memory_cells should be greater than 1')
    no_drift = max_memory_cells == 2

    def in_range(value: int, _range: tuple) -> bool:
        return value >= _range[0] and value < _range[1]

    # no reason for separate logic for now i think?
    def compute_program_size(ranges: list[tuple]) -> int:
        return len(generate_code(ranges))

    def generate_code(ranges: list[tuple]) -> str:
        ranges = ranges.copy()
        n = len(ranges)

        # somehow we sometimes get 1 of the ranges completely unused (usually first or last),
        # and yet it produces a shorter program than n - 1 ranges attempt. Could try looking into that?
        empty_flag = True
        while empty_flag:
            empty_flag = False
            buckets = [ # [[(index, char)], ...] (len(ranges) items)
                [
                    (i, c) for i, c in enumerate(text) if in_range(c, r)
                ] for r in ranges
            ]

            for i in range(len(buckets))[::-1]:
                if len(buckets[i]) == 0:
                    del ranges[i]
                    n = len(ranges)
                    empty_flag = True

        base_values = [b[0][1] if len(b) > 0 else 0 for b in buckets]
        print_sequence = [(i, c, bi) for bi, bucket in enumerate(buckets) for i, c in bucket]
        print_sequence = sorted(print_sequence, key=lambda x: x[0])

        # init base values
        program = string_to_bf_segmented(base_values, no_drift=no_drift, silent=True)
        range_index = n - 1
    
        for _, value, bucket_index in print_sequence:
            shift_delta = bucket_index - range_index
            alt_delta = 4 + min(bucket_index, n - bucket_index - 1)

            # TODO account for this during range selection (esp. range ordering)
            if alt_delta < abs(shift_delta) and not avoid_loops:
                program += f'[{"<" if shift_delta < 0 else ">"}]' + ('>' if shift_delta < 0 else '<') * (alt_delta - 3)
            else:
                program += ('<' if shift_delta < 0 else '>') * abs(shift_delta)

            range_index = bucket_index
            value_delta = value - base_values[bucket_index]
            program += signed_value_to_bf(value_delta)
            base_values[bucket_index] = value
            program += '.'

        c_program = bf_compress(program)
        return c_program

    text_np = np.array(text, dtype=np.int_)

    def generate_ranges_naive(range_count: int) -> list[tuple]:
        mn = np.min(text_np)
        mx = np.max(text_np) + 1
        endpoints = np.linspace(mn, mx, range_count + 1).astype(np.int_).tolist()
        return [(x, y) for x, y in zip(endpoints[:-1], endpoints[1:])]

    def optimize_ranges_naive(ranges: list[tuple]) -> list[tuple]:
        def evaluate_ranges(text_local, ranges):
            range2chars = [text_local[(text_local >= rng[0]) & (text_local < rng[1])] for rng in ranges]
            total_deltas = sum(np.sum(np.abs(chars[1:] - chars[:-1])) for chars in range2chars)

            return total_deltas

        best_ranges = ranges

        for bi in range(len(ranges) - 1):
            test_ranges = best_ranges[:]
            text_np_local = text_np[(text_np >= test_ranges[bi][0]) & (text_np < test_ranges[bi + 1][1])]
            deltas_local = evaluate_ranges(text_np_local, test_ranges[bi:bi+2])
            base_value = test_ranges[bi][1]

            for dir, limit in [(-1, test_ranges[bi][0] + 1), (1, test_ranges[bi + 1][1] - 1)]:
                b_value = base_value

                while b_value != limit:
                    b_value += dir
                    test_ranges[bi] = (test_ranges[bi][0], b_value)
                    test_ranges[bi + 1] = (b_value, test_ranges[bi + 1][1])

                    new_deltas = evaluate_ranges(text_np_local, test_ranges[bi:bi+2])
                    if new_deltas < deltas_local:
                        deltas_local = new_deltas
                        best_ranges = test_ranges[:]

        return best_ranges

    def order_ranges(ranges: list[tuple]) -> list[tuple]:
        n = len(ranges)
        char_order = np.zeros_like(text_np)
        for i, rng in enumerate(ranges[1:], start=1): # technically we don't need to evaluate one of the ranges, let's go with the first one
            char_order[(text_np >= rng[0]) & (text_np < rng[1])] = i

        transition_matrix = np.zeros((n, n), dtype=np.int_).tolist() # TM[x, y] is transition count between range x and range y (both directions)
        for x, y in zip(char_order[:-1], char_order[1:]):
            if x == y:
                continue

            x, y = min(x, y), max(x, y)
            transition_matrix[x][y] += 1

        order_map = [] # (order, transition_cost)

        def get_transition_cost(order: list[int]):
            order_map = [0] * n
            for i, v in enumerate(order):
                order_map[v] = i
            return sum([transition_matrix[x][y] * abs(order_map[x] - order_map[y]) for x in range(n) for y in range(x + 1, n)])

        def compute_order(order: list[int], remaining_options: list[int]):
            if len(remaining_options) == 0:
                order_cost = get_transition_cost(order)
                order_map.append((order, order_cost))
                return

            for opt in remaining_options:
                l_opts = remaining_options[:]
                l_opts.remove(opt)
                compute_order([*order, opt], l_opts)

        def estimate_order():
            def get_partial_cost(order: list[int]):
                order_map = [0] * n
                for i, v in enumerate(order):
                    order_map[v] = i
                return sum([transition_matrix[x][y] * abs(order_map[x] - order_map[y]) for i, x in enumerate(order) for y in order[i + 1:]])

            # [(x, y, c), ...] : c transitions between x and y
            transition_list = [(x, y, transition_matrix[x][y]) for x in range(n) for y in range(x + 1, n)]
            transition_list = sorted(transition_list, key=lambda x: x[2], reverse=True) # highest t-counts are first
            h_order = [*transition_list.pop(0)[0:2]]
            while len(h_order) < n:
                connected_index = next(i for i, t in enumerate(transition_list) if t[0] in h_order or t[1] in h_order)
                x, y, count = transition_list.pop(connected_index)
                missing_range = x if y in h_order else y
                if missing_range in h_order:
                    continue # there are more transitions than ranges

                pre_order = [missing_range, *h_order]
                post_order = [*h_order, missing_range]
                h_order = pre_order if get_partial_cost(pre_order) < get_partial_cost(post_order) else post_order

            order_map.append((h_order, get_transition_cost(h_order)))
            order_map.append((h_order[::-1], get_transition_cost(h_order[::-1])))

        if n <= max_optimal_ranges:
            compute_order([], list(range(n)))
        else:
            estimate_order()

        order_map = sorted(order_map, key=lambda x: x[1])
        best_orders = [x for x in order_map if x[1] == order_map[0][1]]

        best_range_orders = [[ranges[i] for i in order[0]] for order in best_orders]
        return best_range_orders

    best_ranges = [(np.min(text_np), np.max(text_np) + 1)] # [(start_inc; end_exc), ...]; values are char values
    best_size = compute_program_size(best_ranges)
    failed_ranges = 0

    # limit range count to 9 due to O(n!) algo in order_ranges (where n is number of ranges)
    max_range_count = min(max_memory_cells - 2, 256) # 1 memory cell for init factor, 1 for init drift
    if verbose_local > 0:
        print(f'Initialization took {1000 * (perf_counter() - fn_start_time):0.2f}ms. Max range count: {max_range_count}/255')

    for range_count in range(2, max_range_count + 1):
        start_time = perf_counter()
        new_ranges = generate_ranges_naive(range_count)
        new_ranges = optimize_ranges_naive(new_ranges)
        best_range_orders = order_ranges(new_ranges)

        programs = [(compute_program_size(x), x) for x in best_range_orders]
        new_size, new_best_ranges = sorted(programs, key=lambda x: x[0])[0]

        if verbose_local > 1:
            print(f'Attempted {range_count} ranges, best program size: {new_size}. Time elapsed: {1000 * (perf_counter() - start_time):0.2f}ms')
            start_time = perf_counter()

        if new_size < best_size:
            best_size = new_size
            best_ranges = new_best_ranges
            failed_ranges = 0
        elif failed_ranges < 3:
            failed_ranges += 1
        else:
            break

    if verbose_local > 0:
        print(f'Finished trying ranges, best range count: {len(best_ranges)}')

    code_final = generate_code(best_ranges)
    if verbose_local > 0:
        print(f'Total time elapsed: {perf_counter() - fn_start_time:0.2f}s')

    return code_final

def run_self_test(program_generator, args, run_count=250) -> bool:
    print(f'Executing self test: {run_count} runs with args: {args}')

    start_time = perf_counter()
    exception = None
    total_size = 0
    total_encoded_size = 0
    for _ in range(run_count):
        test_input = generate_test_input()
        total_size += len(test_input)
        try:
            test_program = program_generator(test_input, *args)
            total_encoded_size += len(test_program)
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
    print(f'Total input length: {total_size}, total program size: {total_encoded_size}, avg: {total_encoded_size / total_size:0.2f} inst/char')

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

def print_data_bf_inplace(input_data, algo='ranged'):
    if algo == 'basic':
        return bf_print_values_const(input_data)
    if algo == 'ranged':
        return bf_print_values_ranged(input_data)
    else:
        raise Exception(f'Unknown algo: {algo}')

def print_program(program, input_len, no_program, name='default'):
    base_size = len(program)

    if not no_program:
        print(f'Resulting program (algo: {name}): \n\n{program}')

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
    parser.add_argument('--algo', '-a', choices=['segmented', 'perchar', 'basic', 'ranged'], default=[], action='append', help='Conversion algorithm (multiple allowed)')
    parser.add_argument('--mode', '-m', choices=['just-print', 'in-memory'], default='in-memory', help='Program type to generate')
    parser.add_argument('--preset', type=int, help=f'Use a preset string as input. Presets available: {len(input_presets)}')
    parser.add_argument('--no-program', action='store_true', help='Do not output resulting program, only debug messages')
    parser.add_argument('--no-verify', action='store_true', help='Do not verify the generated program for correctness')
    parser.add_argument('--run-test', nargs='?', const=250, default=None, type=int, help='Test algorithm on random inputs, optionally specify number of runs')
    parser.add_argument('--allow-incorrect', action='store_true', help='Print program even if it did not pass verification')
    parser.add_argument('--random-seed', '-s', type=int, default=None, help='Seed to initialize random number generator, e.g. for reproducible test runs')

    args = parser.parse_args()
    verbose = args.verbose

    if args.random_seed is not None:
        print(f'Using custom seed: {args.random_seed}')
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)

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
    elif args.mode == 'just-print':
        algos = args.algo if len(args.algo) > 0 else ['ranged']
        generator = print_data_bf_inplace
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
        program = bf_compress(program)
        if not args.no_verify:
            ok = verify_program(program, input_data)
            if not ok:
                print(f'Error: verification failed for program generated by algo {algo}')
                if not args.allow_incorrect:
                    exit(1)

        # with open('.last_program', 'wt') as tmp_out:
        #     tmp_out.write(program)

        print_program(program, len(input_data), args.no_program, algo)

if __name__ == "__main__":
    main()
