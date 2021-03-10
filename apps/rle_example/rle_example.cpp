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

  IndexVar i("i");

  r0(i) = t0(i) + t0(i);

  r0.setAssembleWhileCompute(true);

  std::cout << r0.getAssignment() << std::endl;
  std::cout << r0.getAssignment().getOperator() << std::endl;
  std::cout << r0.getAssignment().getOperator().defined() << std::endl;

  r0.compile();
  r0.compute();

  std::cout << t0 << std::endl << std::endl;
  std::cout << r0 << std::endl << std::endl;

  Tensor<double> t1("t1", {5}, dv);
}

void matrix(){
  Format  dm({Dense, Dense});
  Format  rlem({Dense, RLE});
  Format  rle2({RLE, RLE});
  IndexVar i,j;

  Tensor<double> t1("t1", {2,3}, dm);
  t1(0,0) = 12;
  t1(0,1) = 12;
  t1(0,2) = 13;
  t1(1,0) = 12;
  t1(1,1) = 12;
  t1(1,2) = 13;

  Tensor<double> r1("r1", {2,3},  rlem);
  Tensor<double> r2("r2", {2,3},  rle2);

  r1(i,j) = t1(i,j);
  r1.setAssembleWhileCompute(true);
  r1.compile();
  r1.compute();

  r2(i,j) = t1(i,j);
  r2.setAssembleWhileCompute(true);
  r2.compile();
  r2.compute();


  std::cout << t1 << std::endl << std::endl;
  std::cout << r1 << std::endl << std::endl;
  std::cout << r2 << std::endl << std::endl;
}

void vector_rle_compute(){
  Format  dv({Dense});
  Format  rlev({RLE});
  IndexVar i;

  Tensor<double> t1("t1", {6}, dv);
  t1(0) = 12;
  t1(1) = 12;
  t1(2) = 13;
  t1(3) = 12;
  t1(4) = 12;
  t1(5) = 13;

  Tensor<double> r1("rFIRST", {6},  rlev);
  Tensor<double> r2("rSECOND", {6},  rlev);

  r1(i) = t1(i);
  r1.setAssembleWhileCompute(true);
  r1.compile();
  r1.compute();

  r2(i) = t1(i);
  r2.setAssembleWhileCompute(true);
  r2.compile();
  r2.compute();

  std::cout << r1 << std::endl << std::endl;
  std::cout << r2 << std::endl << std::endl;


  Tensor<double> t_res("RESULT", {6}, dv);
  t_res(i) = r1(i) + r2(i);
  t_res.evaluate();

//  std::cout << r1 << std::endl << std::endl;
//  std::cout << r2 << std::endl << std::endl;
  std::cout << t_res << std::endl << std::endl;
}


void matrix_rle_compute(){
  Format  dm({Dense, Dense});
  Format  rlem({Dense, RLE});
  IndexVar i,j;

  Tensor<double> t1("t1", {2,3}, dm);
  t1(0,0) = 12;
  t1(0,1) = 12;
  t1(0,2) = 13;
  t1(1,0) = 12;
  t1(1,1) = 12;
  t1(1,2) = 13;

  Tensor<double> r1("rFIRST", {2,3},  rlem);
  Tensor<double> r2("rSECOND", {2,3},  rlem);

  r1(i,j) = t1(i,j);
  r1.setAssembleWhileCompute(true);
  r1.compile();
  r1.compute();

  r2(i,j) = t1(i,j);
  r2.setAssembleWhileCompute(true);
  r2.compile();
  r2.compute();

  std::cout << r1 << std::endl << std::endl;
  std::cout << r2 << std::endl << std::endl;


  Tensor<double> t_res("RESULT", {2,3}, dm);
  t_res(i,j) = r1(i,j) + r2(i,j);
  t_res.evaluate();

  t_res.getTacoTensorT();

//  std::cout << r1 << std::endl << std::endl;
//  std::cout << r2 << std::endl << std::endl;
  std::cout << t_res << std::endl << std::endl;
}

int main(int argc, char* argv[]) {
  vector();
  matrix();
  vector_rle_compute();
//  matrix_rle_compute();
  std::cout << "Done" << std::endl;
}
