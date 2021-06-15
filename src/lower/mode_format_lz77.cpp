//
// Created by Daniel Donenfeld on 6/10/21.
//

#include "taco/lower/mode_format_lz77.h"

#include "ir/ir_generators.h"
#include "taco/ir/simplify.h"
#include "taco/util/strings.h"

using namespace std;
using namespace taco::ir;

namespace taco {

    LZ77ModeFormat::LZ77ModeFormat() :
            LZ77ModeFormat(false, true, true, false) {
    }

    LZ77ModeFormat::LZ77ModeFormat(bool isFull, bool isOrdered,
                                   bool isUnique, bool isZeroless,
                                   long long allocSize) :
            ModeFormatImpl("lz77", isFull, isOrdered, isUnique, false, true,
                           isZeroless, false, true, false, false,
                           true, false, false, false, true, false),
            allocSize(allocSize) {
    }

    ModeFormat LZ77ModeFormat::copy(
            vector<ModeFormat::Property> properties) const {
      bool isFull = this->isFull;
      bool isOrdered = this->isOrdered;
      bool isUnique = this->isUnique;
      bool isZeroless = this->isZeroless;
      for (const auto property : properties) {
        switch (property) {
          case ModeFormat::FULL:
            isFull = true;
            break;
          case ModeFormat::NOT_FULL:
            isFull = false;
            break;
          case ModeFormat::ORDERED:
            isOrdered = true;
            break;
          case ModeFormat::NOT_ORDERED:
            isOrdered = false;
            break;
          case ModeFormat::UNIQUE:
            isUnique = true;
            break;
          case ModeFormat::NOT_UNIQUE:
            isUnique = false;
            break;
          case ModeFormat::ZEROLESS:
            isZeroless = true;
            break;
          case ModeFormat::NOT_ZEROLESS:
            isZeroless = false;
            break;
          default:
            break;
        }
      }
      const auto compressedVariant =
              std::make_shared<LZ77ModeFormat>(isFull, isOrdered, isUnique,
                                                        isZeroless);
      return ModeFormat(compressedVariant);
    }

    ModeFunction LZ77ModeFormat::posIterBounds(Expr parentPos,
                                                        Mode mode) const {
      Expr pbegin = Load::make(getPosArray(mode.getModePack()), parentPos);
      Expr pend = Load::make(getPosArray(mode.getModePack()),
                             ir::Add::make(parentPos, 1));
      Stmt blk =  VarDecl::make(getCoordVar(mode), 0);
      return ModeFunction(blk, {pbegin, pend});
    }

    ModeFunction LZ77ModeFormat::coordBounds(Expr parentPos,
                                                      Mode mode) const {
      taco_not_supported_yet;
      return ModeFunction();
    }



    ModeFunction LZ77ModeFormat::posIterAccess(ir::Expr pos,
                                                        std::vector<ir::Expr> coords,
                                                        Mode mode) const {
      taco_iassert(mode.getPackLocation() == 0);

      Expr posAccessCoord = getPosCoordVar(mode);
      Stmt coordDecl = VarDecl::make(posAccessCoord, getCoordVar(mode));

      Expr distArray = getDistArray(mode.getModePack());
      Expr stride = (int)mode.getModePack().getNumModes();
      Expr dist = Load::make(distArray, ir::Mul::make(pos, stride));

      Expr runArray = getRunArray(mode.getModePack());
      Expr run = Load::make(runArray, ir::Mul::make(pos, stride));

      Stmt ifDistRun = IfThenElse::make( ir::Eq::make(getCoordVar(mode), coords.back()),
              Block::make(IfThenElse::make(ir::Eq::make(dist, 0),
                                        Block::make(addAssign(getCoordVar(mode), 1)),
                                           Block::make(addAssign(getCoordVar(mode), ir::Add::make(run, 1))))));

      Stmt blk = Block::make(coordDecl, ifDistRun);

      return ModeFunction(blk, {posAccessCoord, true});
    }

    ModeFunction LZ77ModeFormat::getFillRegion(ir::Expr pos, std::vector<ir::Expr> coords,
                               Mode mode) const {


      Expr distArray = getDistArray(mode.getModePack());
      Expr stride = (int)mode.getModePack().getNumModes();
      Expr dist = Load::make(distArray, ir::Mul::make(pos, stride));

      Expr runArray = getRunArray(mode.getModePack());
      Expr run = Load::make(runArray, ir::Mul::make(pos, stride));

      Expr updateFill = Var::make("updateFill", Bool);
      Stmt updateFillDecl = VarDecl::make(updateFill, false);
      Stmt ifDistRun = IfThenElse::make(ir::Neq::make(dist, 0),
                                        Assign::make(updateFill, true));

      Expr startPos = ir::Add::make(ir::Sub::make(pos, dist), 1);

      Expr length = Var::make("fill_length", Int());
      Stmt lengthDecl = VarDecl::make(length, Min::make(run,dist));

      Stmt blk = Block::make(updateFillDecl, ifDistRun, lengthDecl);

      return ModeFunction(blk, {startPos, length, run, updateFill});
    }


    vector<Expr> LZ77ModeFormat::getArrays(Expr tensor, int mode,
                                                    int level) const {
      std::string arraysName = util::toString(tensor) + std::to_string(level);
      return {GetProperty::make(tensor, TensorProperty::Indices,
                                level - 1, 0, arraysName + "_pos"),
              GetProperty::make(tensor, TensorProperty::Indices,
                                level - 1, 1, arraysName + "_dist"),
              GetProperty::make(tensor, TensorProperty::Indices,
                                level - 1, 2, arraysName + "_run")};

    }

    Expr LZ77ModeFormat::getPosArray(ModePack pack) const {
      return pack.getArray(0);
    }

    Expr LZ77ModeFormat::getDistArray(ModePack pack) const {
      return pack.getArray(1);
    }

    Expr LZ77ModeFormat::getRunArray(ModePack pack) const {
      return pack.getArray(2);
    }

    Expr LZ77ModeFormat::getCoordVar(Mode mode) const {
      const std::string varName = mode.getName() + "_coord";

      if (!mode.hasVar(varName)) {
        Expr idxCapacity = Var::make(varName, Int());
        mode.addVar(varName, idxCapacity);
        return idxCapacity;
      }

      return mode.getVar(varName);
    }

    Expr LZ77ModeFormat::getPosCoordVar(Mode mode) const {
      const std::string varName = mode.getName() + "_pos_coord";

      if (!mode.hasVar(varName)) {
        Expr idxCapacity = Var::make(varName, Int());
        mode.addVar(varName, idxCapacity);
        return idxCapacity;
      }

      return mode.getVar(varName);
    }


    Expr LZ77ModeFormat::getWidth(Mode mode) const {
      return ir::Literal::make(allocSize, Datatype::Int32);
    }

    bool LZ77ModeFormat::equals(const ModeFormatImpl& other) const {
      return ModeFormatImpl::equals(other) &&
             (dynamic_cast<const LZ77ModeFormat&>(other).allocSize == allocSize);
    }

}
