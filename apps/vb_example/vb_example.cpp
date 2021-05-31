#include <iostream>
#include <taco/lower/mode_format_vb.h>
#include "../../src/lower/iteration_graph.h"
#include "taco.h"

using namespace taco;

ModeFormat LZ(std::make_shared<VariableBlockModeFormat>(false, true, true, false, true));


Index makeVBVectorIndex(const std::vector<int>& rowptr, const std::vector<int>& colidx,
                        const std::vector<int>& szpos) {
  return Index({Compressed, LZ},
               {ModeIndex({makeArray(rowptr), makeArray(colidx)}),
                ModeIndex({makeArray(szpos)})});
}


/// Factory function to construct a compressed sparse row (CSR) matrix.
template<typename T>
TensorBase makeVBVector(const std::string& name, const std::vector<int>& dimensions,
                   const std::vector<int>& pos,
                   const std::vector<int>& crd,
                   const std::vector<int>& szpos,
                   const std::vector<T>& vals) {
  taco_uassert(dimensions.size() == 2) << error::requires_matrix;
  Tensor<T> tensor(name, dimensions, {Compressed, LZ});
  auto storage = tensor.getStorage();
  storage.setIndex(makeVBVectorIndex(pos, crd, szpos));
  storage.setValues(makeArray(vals));
  tensor.setStorage(storage);
  return std::move(tensor);
}

Tensor<double> vb_1(std::string name) {
  return makeVBVector<double>(name, {3,15},
                         {0, 3},
                         {0, 6, 11},
                         {0, 3, 5, 6},
                         {0.0,1.0,2.0,  3.0,4.0, 5.0});
}

int main(int argc, char* argv[]) {
  Format  sv({RLE});
  Format  dv({Dense});
  Format  vb({VB});

  Tensor<double> A("A", {15},   dv, 0);
  Tensor<double> B = vb_1("B");
  Tensor<double> D = vb_1("D");
//  Tensor<double> B("B", {10}, sv, 2);
//  Tensor<double> C("C", {15},     sv,  0);

//
//  // Insert data into B and c
//  C(0) = 4.0;
//  C(8) = 5.0;

  std::cout << B << std::endl;
  std::cout << D << std::endl;

  IndexVar i("i"), B_i_blk("B_i_blk"), D_i_blk("D_i_blk");
  A(i) = B(B_i_blk, i) + D(D_i_blk, i);

  {
    IterationGraph iterationGraph = IterationGraph::make(A.getAssignment());
    iterationGraph.printAsDot(std::cout);
    std::cout << iterationGraph << std::endl;
  }

  A.compile();
  A.printComputeIR(std::cout);
//  A.assemble();
//  A.compute();

//  auto opFunc = [](const std::vector<ir::Expr>& v) {
//    return ir::Add::make(v[0], v[1]);
//  };
//  auto algFunc = [](const std::vector<IndexExpr>& v) {
//    auto l = Region(v[0]);
//    auto r = Region(v[1]);
//    return Union(Union(l, r));
////    return Union(Union(l, r), Union(Complement(l), Complement(r)));
//  };
//  std::initializer_list<Property> properties = {Identity(0), Commutative()};
//
//  Func plus_("plus_", opFunc, algFunc);
//
//  IndexVar i("i"), i_blk("i_blk");
//  A(i) = plus_(B(i_blk, i), C(i));
//
//  A.setAssembleWhileCompute(true);
////  A.evaluate();
//  A.compile();
//  A.printComputeIR(std::cout);
////  A.assemble();
////  A.compute();
//
////  A.printAssembleIR(std::cout);
//
//
////  std::cout << A << std::endl;
}
