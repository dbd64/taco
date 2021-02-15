#include <iostream>
#include "taco.h"

using namespace taco;

void vector() {
  Format dv({Dense});
  Format rlev({RLE});

  Tensor<double> t0("t0", {5}, dv);
  t0(0) = 12;
  t0(1) = 12;
  t0(2) = 12;
  t0(3) = 12;
  t0(4) = 12;

  Tensor<double> r0("r0", {5}, rlev);

  IndexVar i;

  r0(i) = t0(i);

  r0.setAssembleWhileCompute(true);
  r0.evaluate();

  std::cout << t0 << std::endl << std::endl;
  std::cout << r0 << std::endl << std::endl;
}

void matrix(){
  Format  dm({Dense, Dense});
  Format  rlem({Dense, RLE});
  IndexVar i,j;

  Tensor<double> t1("t1", {2,2}, dm);
  t1(0,0) = 12;
  t1(0,1) = 12;
  t1(1,0) = 12;
  t1(1,1) = 12;

  Tensor<double> r1("r1", {2,2},  rlem);

  r1(i,j) = t1(i,j);
  r1.setAssembleWhileCompute(true);
  r1.evaluate();

  std::cout << t1 << std::endl << std::endl;
  std::cout << r1 << std::endl << std::endl;
}

int main(int argc, char* argv[]) {
  vector();
  matrix();
  std::cout << "Done" << std::endl;
}
