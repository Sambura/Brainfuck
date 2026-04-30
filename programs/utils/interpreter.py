from time import perf_counter
from collections.abc import Iterable, Callable
from typing import Any
from itertools import takewhile

def format_elapsed_time(seconds):
    if seconds > 150:
        return f'{seconds // 60}m {seconds % 60}s'
    if seconds >= 2:
        return f'{seconds:0.3f}s'
    if seconds > 0.002:
        return f'{1000 * seconds:0.2f}ms'
    
    return f'{1000000 * seconds:0.3f}us'

def _bf_execute_basic(program: str, input_generator: Iterable[int], output_consumer: Callable[[int], None],
                      iter_limit:int=None, **kwargs) -> None:
    """ Execute brainfuck program.
        Simplest interpreter: works directly on a string with the program   """

    if len(kwargs) > 0:
        print(f'Warning: ignoring unknown kwargs {kwargs}')

    memory = [0] * 32768
    pc = 0
    ptr = 0
    loop_stack = []
    loop_iters = 0 # count only loop iterations to save on performance
    if iter_limit is None: iter_limit = 50000000

    while pc < len(program):
        c = program[pc]

        if c == '+':
            memory[ptr] = (memory[ptr] + 1) & 0xFF
        elif c == '-':
            memory[ptr] = (memory[ptr] - 1) & 0xFF
        elif c == '>':
            ptr += 1
        elif c == '<':
            ptr -= 1
            if ptr < 0:
                raise Exception('Memory underflow')
        elif c == '.':
            output_consumer(memory[ptr])
        elif c == ',':
            memory[ptr] = next(input_generator) & 0xFF
        elif c == '[':
            if memory[ptr] != 0:
                loop_stack.append(pc)
            else:
                nest_level = 1
                while nest_level > 0:
                    cc = program[pc + 1]
                    if cc == '[':
                        nest_level += 1
                    elif cc == ']':
                        nest_level -= 1
                    pc += 1
        elif c == ']':
            pc = loop_stack.pop() - 1
            loop_iters += 1
            if loop_iters >= iter_limit:
                raise Exception('Iteration limit reached')

        pc += 1

def _bf_execute_preprocessed(program: str, input_generator: Iterable[int], output_consumer: Callable[[int], None],
                             iter_limit: int=None, control_char=None, optimization_level=2, **kwargs) -> None:
    """ Execute brainfuck program.
        Processes program into list of opcodes before executing.
        Also combines consecutive +/- and </> instructions          """
    
    if len(kwargs) > 0:
        print(f'Warning: ignoring unknown kwargs {kwargs}')

    # OPCODES:
    #   * Base opcodes: 0 - NOOP, 1 - ADD, 2 - SHIFT, 3 - BEGIN_LOOP, 4 - END_LOOP, 5 - PRINT, 6 - GET
    #   * Control opcode: 7 - CONTROL (custom command)
    #   * Optimization opcodes: 8 - LOAD IMMEDIATE, 9 - MULTI SUM, 10 - INVERT, 11 - GLIDE, 12 - OFFSET ADD

    def preprocess(program: str):
        opcodes = [0]
        consts = [0]
        control_dict = {}
        loop_stack = []
        skip_flag = False

        for i, c in enumerate(program):
            if skip_flag:
                skip_flag = False
                continue

            if c == '+' or c == '-':
                value = 1 if c == '+' else -1
                if opcodes[-1] == 1 and optimization_level > 0:
                    consts[-1] += value
                else:
                    opcodes.append(1)
                    consts.append(value)
            elif c == '<' or c == '>':
                value = 1 if c == '>' else -1
                if opcodes[-1] == 2 and optimization_level > 0:
                    consts[-1] += value
                else:
                    opcodes.append(2)
                    consts.append(value)
            elif c == '.':
                opcodes.append(5)
                consts.append(0) # we *could* count dots, but if would probably just hurt runtime performance, since '.' is almost never repeated
            elif c == ',':
                opcodes.append(6)
                consts.append(0) # same as with '.'
            elif c == '[':
                loop_stack.append(len(opcodes)) # PC + 1
                opcodes.append(3)
                consts.append(0)
            elif c == ']':
                if len(loop_stack) == 0:
                    raise Exception(f'Brainfuck syntax error: unexpected "]" at index {i}')

                loop_start = loop_stack.pop()
                past_loop_end = len(opcodes)
                opcodes.append(4)
                consts.append(loop_start - 1) # PC for BEGIN_LOOP
                consts[loop_start] = past_loop_end - 1 # PC for END_LOOP
            elif c == control_char:
                subcommand = program[i + 1] if len(program) > i + 1 else None

                def get_message(start_index):
                    message_indices = list(takewhile(lambda x: program[x] not in '+-[],.<>\n\r$', range(start_index, len(program))))
                    message_end = message_indices[-1] if len(message_indices) > 0 else start_index - 1
                    return program[start_index:message_end + 1]

                if subcommand == '%': # measure time
                    skip_flag = subcommand == control_char
                    control_dict[len(opcodes) - 1] = get_message(i + 2)
                    consts.append(1)
                else:
                    control_dict[len(opcodes) - 1] = get_message(i + 1)
                    consts.append(0)

                opcodes.append(7)
        
        if len(loop_stack) > 0:
            raise Exception(f'Brainfuck syntax error: unmatched "["')

        return opcodes[1:], consts[1:], control_dict

    # TODO: implement the thing that will prevent optimization if it will consume control opcode
    def optimize(opcodes: list[int], consts: list[int], control_dict: dict[int, dict]):
        ##### Step 1: decouple logic from control #####
        program_logic = [] # (begin_id, end_id, opcode, optim_const, [TODO: has_bound_control])
        program_control = [] # (id, control_opcode)
        pre_ids = list(range(len(opcodes)))
        __last_id = len(opcodes) - 1
        __verbose = False #or True

        def alloc_id():
            nonlocal __last_id
            __last_id += 1
            return __last_id

        for id, opcode, const in zip(pre_ids, opcodes, consts):
            if opcode == 7:
                program_control.append((id, const))
                continue

            program_logic.append((id, id, opcode, const))

        ##### Step 2: optimize logic #####
        def detect_load_immediate(index: int, succeeding: int) -> tuple[int, int, int]: # (start inclusive, end inclusive, const)
            if succeeding < 2 or program_logic[index + 1][2] != 1 or program_logic[index + 2][2] != 4 or program_logic[index + 1][3] % 2 == 0:
                return

            if __verbose: print(f'Detected LI at {index}')
            return index, index + 2, 0

        def absorb_load_immediate(index: int, succeeding: int) -> tuple[int, int, int]:
            start_index = index - 1 if index > 0 and program_logic[index - 1][2] == 1 else index # adds leading to LI are useless, remove
            have_si = succeeding > 0 and program_logic[index + 1][2] == 1
            succeeding_increment = program_logic[index + 1][3] if have_si else 0

            if start_index != index or have_si:
                if __verbose: print(f'Detected LIA at {index}')
                return start_index, index + 1 if have_si else index, program_logic[index][3] + succeeding_increment

        def detect_multi_sum(index: int, succeeding: int) -> tuple[int, int, Any]: # triggered on begin loop
            # minimal multi sum: [->+<] (6 instructions) (this is actually an example of mono sum)
            loop_code = program_logic[index + 1:list(takewhile(lambda x: program_logic[x][2] != 4, range(index + 1, len(program_logic))))[-1] + 1]

            if len(loop_code) < 4 or len([x for x in loop_code if x[2] != 1 and x[2] != 2]) > 0: # drop if loop has anything other than add or shift
                return

            increment_profile = {} # { offset: increment_amount, ...]
            offset = 0
            for _, _, opcode, const in loop_code:
                if opcode == 1: # ADD
                    current_increment = increment_profile.get(offset, 0)
                    increment_profile[offset] = current_increment + const
                elif opcode == 2: # SHIFT
                    offset += const
                else:
                    assert False

            if 0 not in increment_profile or abs(increment_profile[0]) != 1:
                return # if offset 0 is incremented by something other than 1 or -1 we cannot guarantee that the loop terminates, figure out how to do it later
            do_invert = increment_profile.pop(0) == 1 # if we increment current value instead of decrementing, just invert the value before looping
            offsets = sorted(list(increment_profile.items()), key=lambda x: x[0])
            assert len(offsets) > 0
            new_instructions = [(10, 0), (9, offsets)] if do_invert else [(9, offsets)]
            if __verbose: print(f'Detected MS#{len(offsets)} at {index}')

            return index, index + len(loop_code) + 1, new_instructions

        def detect_glider(index: int, succeeding: int) -> tuple[int, int, int]:
            if succeeding < 2 or program_logic[index + 1][2] != 2 or program_logic[index + 2][2] != 4:
                return

            if __verbose: print(f'Detected GL at {index}')
            return index, index + 2, program_logic[index + 1][3]

        def detect_offset_add(index: int, succeeding: int) -> tuple[int, int, Any]:
            if succeeding < 2 or program_logic[index + 1][2] != 1 or program_logic[index + 2][2] != 2 or program_logic[index][3] != -program_logic[index + 2][3]:
                return

            if __verbose: print(f'Detected OADD at {index}')
            return index, index + 2, (program_logic[index][3], program_logic[index + 1][3])

        optimizers = [
            # (trigger_opcode, detector -> const, new_opcode)
            (3, detect_load_immediate, 8),
            (8, absorb_load_immediate, 8),
            (3, detect_multi_sum, None),
            (3, detect_glider, 11),
            (2, detect_offset_add, 12), # if optimizers are run one by one, this one should be run after multi sum one
        ]

        reiterate_flag = True
        while reiterate_flag:
            reiterate_flag = False
            o_start, o_end, new_instructions = None, None, []

            for i, (_, _, opcode, _) in enumerate(program_logic):
                for trigger_opcode, optimizer, new_opcode in optimizers:
                    if trigger_opcode != opcode:
                        continue

                    result = optimizer(i, len(program_logic) - i - 1)
                    if result is None:
                        continue

                    o_start, o_end, new_instructions = result
                    if new_opcode is not None: # treat new_instructions as a const value for this opcode
                        new_instructions = [(new_opcode, new_instructions)]

                    break

                if len(new_instructions) > 0:
                    reiterate_flag = True
                    source_slice = program_logic[o_start:o_end + 1]
                    begin_id, end_id = source_slice[0][0], source_slice[-1][1]
                    id_sequence = [begin_id] + [alloc_id() for _ in range(len(new_instructions) - 1)] + [end_id]
                    new_logic = []

                    for i, (opcode, const) in enumerate(new_instructions):
                        new_logic.append((id_sequence[i], id_sequence[i + 1], opcode, const))

                    program_logic = program_logic[:o_start] + new_logic + program_logic[o_end + 1:]

                    break

        ##### Step 3: inject controls back into program logic #####
        for id, control_opcode in program_control:
            control_unit = (id, id, 7, control_opcode)
            control_index = 0

            if id != 0:
                target_units = [i for i, x in enumerate(program_logic) if x[1] == id - 1]
                if len(target_units) == 0:
                    print(f'Warning: could not re-inject control #{id}. This control opcode will be ignored. Use -O0 or -O1 to fix')
                    continue

                control_index = target_units[0] + 1

            program_logic.insert(control_index, control_unit)

        ##### Step 4: convert to interpreter format #####
        new_opcodes = []
        new_consts = []
        def id2opcode(id):
            return next((i for i, x in enumerate(program_logic) if x[0] == id), None)

        for id1, id2, opcode, const in program_logic:
            new_opcodes.append(opcode)
            if opcode == 3 or opcode == 4: # loop opcodes
                # current const: logic unit's ID. need to convert that into opcode index
                const = id2opcode(const)

            new_consts.append(const)

        ##### Step 5: rebuild control dict #####
        new_control_dict = {}
        for id, value in control_dict.items():
            opcode = id2opcode(id)

            if opcode is not None: # true except if control was not re-injected
                new_control_dict[opcode] = value

        return new_opcodes, new_consts, new_control_dict

    opcodes, consts, control_dict = preprocess(program)

    if optimization_level > 1:
        opcodes, consts, control_dict = optimize(opcodes, consts, control_dict)

    memory = [0] * 32768
    pc = 0
    ptr = 0
    loop_iters = 0
    ctl_timestamps = {}
    if iter_limit is None: iter_limit = 100000000

    while pc < len(opcodes):
        opcode = opcodes[pc]

        if opcode == 1: # ADD
            memory[ptr] = (memory[ptr] + consts[pc]) & 0xFF
        elif opcode == 2: # SHIFT
            ptr += consts[pc]
            if ptr < 0:
                raise Exception('Memory underflow')
        elif opcode == 4 and memory[ptr] != 0: # END_LOOP
            pc = consts[pc]
            loop_iters += 1
            if loop_iters >= iter_limit:
                raise Exception('Iteration limit reached')
        elif opcode == 3 and memory[ptr] == 0: # BEGIN_LOOP
            pc = consts[pc]
        elif opcode == 5:   # PRINT
            output_consumer(memory[ptr])
        elif opcode == 6:   # GET
            memory[ptr] = next(input_generator) & 0xFF
        elif opcode == 8:   # LOAD IMMEDIATE
            memory[ptr] = consts[pc] & 0xFF
        elif opcode == 9:   # MULTI SUM
            offsets = consts[pc]    # offsets should be sorted (ascending)
            value = memory[ptr]
            if ptr + offsets[0][0] < 0:
                raise Exception('Memory underflow')
            memory[ptr] = 0         # current ptr is zeroed out by default. Offset 0 will cancel this out, if present
            for offset, factor in offsets:
                c_ptr = ptr + offset
                memory[c_ptr] = (memory[c_ptr] + value * factor) & 0xFF
        elif opcode == 12:  # OFFSET SUM
            offset, increment = consts[pc]
            c_ptr = ptr + offset
            if c_ptr < 0:
                raise Exception('Memory underflow')
            memory[c_ptr] = (memory[c_ptr] + increment) & 0xFF
        elif opcode == 11:  # GLIDER
            step = consts[pc]
            while memory[ptr] != 0:
                ptr += step
                if ptr < 0:
                    raise Exception('Memory underflow')
        elif opcode == 10:  # INVERT
            c_ptr = ptr + consts[pc]
            if c_ptr < 0:
                raise Exception('Memory underflow')
            memory[c_ptr] = (256 - memory[c_ptr]) & 0xFF
        elif opcode == 7:   # CONTROL
            if consts[pc] == 0: # CONTROL:0 state dump
                print(f'\n[STATE DUMP]: {control_dict[pc]}. Program counter: {pc}, memory pointer: {ptr}. Memory:\n')
                last_nonzero_index = [i for i, m in enumerate(memory) if m != 0][-1]
                slice_start = 0
                slice_width = 20
                while slice_start - 1 < max(last_nonzero_index, ptr + 1):
                    current_start = slice_start
                    memory_slice = memory[slice_start:slice_start + slice_width]
                    slice_start += slice_width
                    string_slice = ' '.join(f'{m:3}' for m in memory_slice)
                    print(f'[{current_start:5}:{slice_start - 1:5}]: {string_slice}')
            elif consts[pc] == 1: # CONTROL:1 measure time
                timestamp = perf_counter()
                ts_name = control_dict[pc]
                last_timestamp = ctl_timestamps.get(ts_name, None)
                if last_timestamp is not None:
                    print(f'\n[TIME MEASURE]: {ts_name}: {format_elapsed_time(timestamp - last_timestamp)} elapsed\n')
                ctl_timestamps[ts_name] = perf_counter()

        pc += 1

def _bf_get_interpreter(implementation: str='best'):
    implementation = implementation.lower()
    if implementation == 'best' or implementation == 'preprocessed':
        return _bf_execute_preprocessed
    elif implementation == 'basic':
        return _bf_execute_basic
    else:
        raise Exception(f'Invalid implementation name: "{implementation}"')

def _bf_execute(program: str, in_data: list[int], interpreter: Callable, **kwargs) -> list[int]:
    stdout = []

    def input_generator():
        yield from in_data
        while True:
            yield 0

    interpreter(program, input_generator(), stdout.append, **kwargs)

    return stdout

def _bf_interactive_buffered_execute(program: str, interpreter: Callable, input_prompt: bool=False, **kwargs) -> float:
    io_time = 0
    last_char = None

    def input_generator():
        nonlocal io_time, last_char

        while True:
            start_time = perf_counter()
            prompt = ('$> ' if last_char == '\n' else '\n$> ') if input_prompt else ''
            last_char = '\n'
            buffer = list(map(ord, input(prompt).encode().decode('unicode_escape'))) + [ord('\n')]
            io_time += perf_counter() - start_time
            yield from buffer

    def output_consumer(char: int):
        nonlocal last_char
        last_char = char
        print(chr(char), end='')

    interpreter(program, input_generator(), output_consumer, **kwargs)

    return io_time

def bf_execute(program: str, in_data: list[int]=[], implementation='best', **kwargs) -> list[int]:
    "Execute a brainfuck program with a given input. Returns data printed to stdout by the program"

    return _bf_execute(program, in_data, _bf_get_interpreter(implementation), **kwargs)

def bf_interactive_buffered_execute(program: str, implementation='best', input_prompt: bool=False, **kwargs) -> float:
    "Execute a brainfuck program interactively. Input is sent to the program after user presses Enter. Returns time spent waiting for user input"

    return _bf_interactive_buffered_execute(program, _bf_get_interpreter(implementation), input_prompt=input_prompt, **kwargs)

def main():
    import argparse

    # make a parser
    parser = argparse.ArgumentParser(description='Interpret a brainfuck program')

    parser.add_argument('program', nargs='?', default=None, help='Program to interpret')
    parser.add_argument('--file', '-f', type=str, help='Take program from a file instead')
    parser.add_argument('--implementation', type=str, default='best', choices=['best', 'basic', 'preprocessed'], help='Interpreter implementation name to use')
    parser.add_argument('--iter-limit', '-l', type=int, default=-1, help='Limit iteration count performed by interpreter (only "]" instruction counts as iteration)')
    parser.add_argument('--control-char', '-c', type=str, nargs='?', const='$', default=None, help='Character to use for debug interpreter commands like state dumping or time measurement')
    parser.add_argument('-O', type=int, default=2, help='Optimization level. Currently supports values 0, 1 or 2')
    parser.add_argument('--raise', '-r', action='store_true', default=False, help='Upon interpreter error, print exception with backtrace')

    # parse and validate arguments
    args = parser.parse_args()
    if args.iter_limit is not None and args.iter_limit < 0:
        args.iter_limit = float('inf')

    if args.control_char is not None:
        if args.implementation == 'basic':
            print('Error: Basic interpreter does not support control characters')
            exit(1)
        if len(args.control_char) != 1:
            print('Error: Expected a single character as a control character argument')
            exit(1)
        if args.control_char in '+-[]<>,.':
            print(f'Error: Cannot use {args.control_char} as a control character since it is an instruction character')
            exit(1)

    if args.program and args.file:
        print('Error: Multiple program input methods')
        exit(1)

    program = args.program
    if args.file is not None:
        try:
            with open(args.file, mode='rt') as file:
                program = file.read()
                print(f'Read program from {args.file}. {len(program)} bytes read')
        except Exception as e:
            print(f'Error: failed to read file {args.file}: {e}')
            exit(1)
    elif program is None:
        program = input("Input a brainfuck program: ")

    if args.O > 2:
        print(f'Warning: optimization level {args.O} not supported, using highest available instead')
        args.O = 2
    elif args.O < 0:
        print(f'Error: Invalid optimization level {args.O}')
        exit(1)

    # start interpreting
    start_time = perf_counter()
    try:
        print('[Started program execution]')
        io_time = bf_interactive_buffered_execute(program, args.implementation, iter_limit=args.iter_limit, control_char=args.control_char, optimization_level=args.O)
    except Exception as e:
        io_time = 0
        print(f'\n[Encountered error during execution: {e}. Terminating]')

        if getattr(args, 'raise'): # same as args.raise, avoids syntax error
            raise e
        exit(1)
    except KeyboardInterrupt:
        print(f'\n[Interrupted. Was executing for {perf_counter() - start_time:0.2f}s]')
        exit(1)

    print(f'\n[Program terminated. Execution took {perf_counter() - start_time - io_time:0.2f}s. I/O time: {io_time:0.2f}s]')

if __name__ == '__main__':
    main()
