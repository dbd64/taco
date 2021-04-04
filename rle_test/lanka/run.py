import subprocess
import os
from functools import reduce

REPEAT = 200
TACO_TOOL = "/Users/danieldonenfeld/Developer/taco/cmake-build-release-lanka/bin/taco"
OUT_CSV = "out.csv"

def append_line(line):
  with open(OUT_CSV, "a") as file:
    file.write(line + "\n")

def write_csv_header():
  prefix = "{},{},{},{},{},{},{},".format("rle_low", "rle_high", "value_low", "value_high","size","bits", "tensor_num")

  acc = reduce(lambda acc,x: acc + str(x) if acc == "" else acc + "," + str(x), range(REPEAT- 2*int(REPEAT*0.1)), "") 

  with open(OUT_CSV, "w") as file:
    file.write(prefix+acc + "\n")

def run_taco_tool():
  mode_format = 'r' if bits == 16 else ('y' if bits == 8 else 't')
  args = [TACO_TOOL, 't(i) = f(i)+ s(i)', '-f=t:{}'.format(mode_format), '-f=f:{}'.format(mode_format),\
     '-f=s:{}'.format(mode_format), '-c', '-write-source=vec_rle.c']
  out = subprocess.run(args, check=True)
  print(out)


def compile_benchmark(rle_range, value_range, size,  bits):
  print("Compiling: {}, {}, {}, {}".format(rle_range, value_range, size, bits))
  # First need to run the taco tool to generate the code
  run_taco_tool()

  additional_cmake_args = ["-DSIZE={}".format(size), "-DREPEAT={}".format(REPEAT),\
    "-DLOWER_RLE={}".format(rle_range[0]), "-DUPPER_RLE={}".format(rle_range[1]),\
    "-DUPPER_VAL={}".format(value_range[1]), "-DRLE_BITS={}".format(bits)]

  # Next use cmake to generate the benchmark binary
  # os.environ['TACO_INCLUDE_DIR'] = "/Users/danieldonenfeld/Developer/taco/include"
  # os.environ['TACO_LIBRARY_DIR'] = "/Users/danieldonenfeld/Developer/taco/cmake-build-release-lanka/lib"
  subprocess.run(["rm", "-r", "TMP_BUILD"])
  subprocess.run(["mkdir", "TMP_BUILD"], check=True)
  subprocess.run(["cmake", "-DCMAKE_BUILD_TYPE=Release"] + additional_cmake_args + ['..'], check=True, cwd="TMP_BUILD")
  # "/Users/danieldonenfeld/Library/Application Support/JetBrains/Toolbox/apps/CLion/ch-0/203.5981.166/CLion.app/Contents/bin/cmake/mac/bin/cmake" --build /Users/danieldonenfeld/Developer/taco/cmake-build-release-lanka --target manual
  subprocess.run(["cmake", "--build", ".", "--target", "rle_bench_vec"], check=True, cwd="TMP_BUILD")


def process_raw_data(line):
  data = [float(x) for x in line[:-1].split(",")]
  data.sort()
  trunc = int(len(data)*0.1)
  data = data[trunc:-trunc]
  return data

def append_row(rle_range, value_range, size,  bits,tensor_num, data):
  prefix = "{},{},{},{},{},{},{},".format(rle_range[0], rle_range[1],value_range[0], value_range[1],size,bits,tensor_num)
  acc= ""; [acc := acc + str(x) if acc == "" else acc + "," + str(x) for x in data]
  acc = reduce(lambda acc,x: acc + str(x) if acc == "" else acc + "," + str(x), data, "") 

  append_line(prefix + acc)

def run_benchmark(rle_range, value_range, size,  bits, tensor_num):
  # Run the benchmark binary
  print("Running: {},{},{},{},{},{},{}".format(rle_range[0], rle_range[1],value_range[0], value_range[1],size,bits,tensor_num))
  result = subprocess.run(["./rle_bench_vec"], check=True, capture_output=True, text=True, cwd="TMP_BUILD")

  # Parse and Store results
  lines = result.stdout.split("\n")
  rle_data = process_raw_data(lines[0])
  dense_data = process_raw_data(lines[1])

  append_row(rle_range, value_range, size, bits,tensor_num, rle_data)
  append_row(rle_range, value_range, size, 0,tensor_num, dense_data)

def cleanup():
  # Clean up 
  # subprocess.run(["rm", "vec_rle.c"], check=True)
  pass

# rle_ranges = [(1, 10), (1, 1000), (1000, 10000)]
# sizes  = [10**x for x in range(2,4)]
# values = [(0,10**x) for x in range(0,2)]
# bits = [2**x for x in range(3,6)]
# numRandTensors = 1;

rle_ranges = [(1000, 10000)]
sizes  = [10**x for x in range(2,3)]
values = [(0,10**x) for x in range(0,1)]
bits = [2**x for x in range(3,4)]
numRandTensors = 2;

write_csv_header()
for rle_range in rle_ranges:
  for size in sizes:
    for value_range in values:
      for bit in bits:
        compile_benchmark(rle_range, value_range, size,  bit)
        for i in range(numRandTensors):
          run_benchmark(rle_range, value_range, size,  bit, i)
        
        cleanup()