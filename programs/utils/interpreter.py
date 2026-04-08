from time import perf_counter
from collections.abc import Iterable, Callable
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
                             iter_limit: int=None, control_char=None, **kwargs) -> None:
    """ Execute brainfuck program.
        Processes program into list of opcodes before executing.
        Also combines consecutive +/- and </> instructions          """
    
    if len(kwargs) > 0:
        print(f'Warning: ignoring unknown kwargs {kwargs}')

    def preprocess(program: str):
        opcodes = [0] # 0 - NOOP, 1 - ADD, 2 - SHIFT, 3 - BEGIN_LOOP, 4 - END_LOOP, 5 - PRINT, 6 - GET, 7 - CONTROL (custom command)
        consts = [0]
        control_dict = {0: {}, 1: {}}
        loop_stack = []
        skip_flag = False

        for i, c in enumerate(program):
            if skip_flag:
                skip_flag = False
                continue

            if c == '+' or c == '-':
                value = 1 if c == '+' else -1
                if opcodes[-1] == 1:
                    consts[-1] += value
                else:
                    opcodes.append(1)
                    consts.append(value)
            elif c == '<' or c == '>':
                value = 1 if c == '>' else -1
                if opcodes[-1] == 2:
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
                    message_end = message_indices[-1] if len(message_indices) > 0 else start_index
                    return program[start_index:message_end + 1]

                if subcommand == '%': # measure time
                    skip_flag = subcommand == control_char
                    message = get_message(i + 2)
                    prev_record = control_dict[1].get(message, ([], None)) # ([opcode indices], last-measurement)
                    prev_record[0].append(len(opcodes) - 1)
                    control_dict[1][message] = prev_record
                    consts.append(1)
                else:
                    control_dict[0][len(opcodes) - 1] = get_message(i + 1)
                    consts.append(0)

                opcodes.append(7)
        
        if len(loop_stack) > 0:
            raise Exception(f'Brainfuck syntax error: unmatched "["')

        return opcodes[1:], consts[1:], control_dict

    opcodes, consts, control_dict = preprocess(program)
    memory = [0] * 32768
    pc = 0
    ptr = 0
    loop_iters = 0
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
        elif opcode == 7:   # CONTROL
            if consts[pc] == 0: # CONTROL:0 state dump
                print(f'\n[STATE DUMP]: {control_dict[0][pc]}. Program counter: {pc}, memory pointer: {ptr}. Memory:\n')
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
                for message, record in control_dict[1].items():
                    if pc in record[0]:
                        last_timestamp = record[1]
                        if last_timestamp is not None:
                            print(f'\n[TIME MEASURE]: {message}: {format_elapsed_time(timestamp - last_timestamp)} elapsed\n')
                        control_dict[1][message] = (record[0], perf_counter())
                        break

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

    # start interpreting
    start_time = perf_counter()
    try:
        print('[Started program execution]')
        io_time = bf_interactive_buffered_execute(program, args.implementation, iter_limit=args.iter_limit, control_char=args.control_char)
    except Exception as e:
        io_time = 0
        print(f'\n[Encountered error during execution: {e}. Terminating]')
        exit(1)
    except KeyboardInterrupt:
        print(f'\n[Interrupted. Was executing for {perf_counter() - start_time:0.2f}s]')
        exit(1)

    print(f'\n[Program terminated. Execution took {perf_counter() - start_time - io_time:0.2f}s. I/O time: {io_time:0.2f}s]')

if __name__ == '__main__':
    main()
