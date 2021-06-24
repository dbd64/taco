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
                           true, false, false, false, true,
                           true, false),
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

    Stmt LZ77ModeFormat::getAppendCoord(Expr p, Expr i, Mode mode) const {
      taco_iassert(mode.getPackLocation() == 0);
      Expr distArray = getDistArray(mode.getModePack());
      Expr runArray = getRunArray(mode.getModePack());
      Expr stride = (int)mode.getModePack().getNumModes();
      Expr idx = ir::Mul::make(p, stride);


      Stmt maybeResizeDist = doubleSizeIfFull(distArray, getDistCapacity(mode), p);
      Stmt maybeResizeRun = doubleSizeIfFull(runArray, getRunCapacity(mode), p);


      return Block::make( maybeResizeDist, maybeResizeRun,
              Store::make(distArray, idx, 0),
              Store::make(runArray, idx, 0));
    }

    Stmt LZ77ModeFormat::getAppendEdges(Expr pPrev, Expr pBegin, Expr pEnd,
                                              Mode mode) const {
      Expr posArray = getPosArray(mode.getModePack());
      ModeFormat parentModeType = mode.getParentModeType();
      Expr edges = (!parentModeType.defined() || parentModeType.hasAppend())
                   ? pEnd : ir::Sub::make(pEnd, pBegin);
      return Store::make(posArray, ir::Add::make(pPrev, 1), edges);
    }

    Expr LZ77ModeFormat::getSize(ir::Expr szPrev, Mode mode) const {
      return Load::make(getPosArray(mode.getModePack()), szPrev);
    }

    Stmt LZ77ModeFormat::getAppendInitEdges(Expr pPrevBegin,
                                                  Expr pPrevEnd, Mode mode) const {
      if (isa<ir::Literal>(pPrevBegin)) {
        taco_iassert(to<ir::Literal>(pPrevBegin)->equalsScalar(0));
        return Stmt();
      }

      Expr posArray = getPosArray(mode.getModePack());
      Expr posCapacity = getPosCapacity(mode);
      ModeFormat parentModeType = mode.getParentModeType();
      if (!parentModeType.defined() || parentModeType.hasAppend()) {
        return doubleSizeIfFull(posArray, posCapacity, pPrevEnd);
      }

      Expr pVar = Var::make("p" + mode.getName(), Int());
      Expr lb = ir::Add::make(pPrevBegin, 1);
      Expr ub = ir::Add::make(pPrevEnd, 1);
      Stmt initPos = For::make(pVar, lb, ub, 1, Store::make(posArray, pVar, 0));
      Stmt maybeResizePos = atLeastDoubleSizeIfFull(posArray, posCapacity, pPrevEnd);
      return Block::make({maybeResizePos, initPos});
    }

    Stmt LZ77ModeFormat::getAppendInitLevel(Expr szPrev, Expr sz,
                                                  Mode mode) const {
      const bool szPrevIsZero = isa<ir::Literal>(szPrev) &&
                                to<ir::Literal>(szPrev)->equalsScalar(0);

      Expr defaultCapacity = ir::Literal::make(allocSize, Datatype::Int32);
      Expr posArray = getPosArray(mode.getModePack());
      Expr initCapacity = szPrevIsZero ? defaultCapacity : ir::Add::make(szPrev, 1);
      Expr posCapacity = initCapacity;

      std::vector<Stmt> initStmts;
      if (szPrevIsZero) {
        posCapacity = getPosCapacity(mode);
        initStmts.push_back(VarDecl::make(posCapacity, initCapacity));
      }
      initStmts.push_back(Allocate::make(posArray, posCapacity));
      initStmts.push_back(Store::make(posArray, 0, 0));

      if (mode.getParentModeType().defined() &&
          !mode.getParentModeType().hasAppend() && !szPrevIsZero) {
        Expr pVar = Var::make("p" + mode.getName(), Int());
        Stmt storePos = Store::make(posArray, pVar, 0);
        initStmts.push_back(For::make(pVar, 1, initCapacity, 1, storePos));
      }

      initStmts.push_back(Allocate::make(getDistArray(mode.getModePack()), 100));
      initStmts.push_back(Allocate::make(getRunArray(mode.getModePack()), 100));
      initStmts.push_back(VarDecl::make(getDistCapacity(mode), 100));
      initStmts.push_back(VarDecl::make(getRunCapacity(mode), 100));

      return Block::make(initStmts);
    }

    Stmt LZ77ModeFormat::getAppendFinalizeLevel(Expr szPrev,
                                                      Expr sz, Mode mode) const {
      ModeFormat parentModeType = mode.getParentModeType();
      if ((isa<ir::Literal>(szPrev) && to<ir::Literal>(szPrev)->equalsScalar(1)) ||
          !parentModeType.defined() || parentModeType.hasAppend()) {
        return Stmt();
      }

      Expr csVar = Var::make("cs" + mode.getName(), Int());
      Stmt initCs = VarDecl::make(csVar, 0);

      Expr pVar = Var::make("p" + mode.getName(), Int());
      Expr loadPos = Load::make(getPosArray(mode.getModePack()), pVar);
      Stmt incCs = Assign::make(csVar, ir::Add::make(csVar, loadPos));
      Stmt updatePos = Store::make(getPosArray(mode.getModePack()), pVar, csVar);
      Stmt body = Block::make({incCs, updatePos});
      Stmt finalizeLoop = For::make(pVar, 1, ir::Add::make(szPrev, 1), 1, body);

      return Block::make({initCs, finalizeLoop});
    }

    Stmt
    LZ77ModeFormat::getFillRegionAppend(ir::Expr p, ir::Expr i,
                                        ir::Expr start, ir::Expr length,
                                        ir::Expr run, Mode mode) const {
      Expr distArray = getDistArray(mode.getModePack());
      Expr stride = (int)mode.getModePack().getNumModes();
      Expr distValue = ir::Sub::make(p, start);
      Stmt storeDist = Store::make(distArray, ir::Mul::make(p, stride), distValue);

      Expr runArray = getRunArray(mode.getModePack());
      Stmt storeRun = Store::make(runArray, ir::Mul::make(p, stride), run);

      return Block::make(storeDist, storeRun);
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

    Expr LZ77ModeFormat::getPosCapacity(Mode mode) const {
      const std::string varName = mode.getName() + "_pos_capacity";

      if (!mode.hasVar(varName)) {
        Expr posCapacity = Var::make(varName, Int());
        mode.addVar(varName, posCapacity);
        return posCapacity;
      }

      return mode.getVar(varName);
    }

    Expr LZ77ModeFormat::getRunCapacity(Mode mode) const {
      const std::string varName = mode.getName() + "_run_capacity";

      if (!mode.hasVar(varName)) {
        Expr posCapacity = Var::make(varName, Int());
        mode.addVar(varName, posCapacity);
        return posCapacity;
      }

      return mode.getVar(varName);
    }

    Expr LZ77ModeFormat::getDistCapacity(Mode mode) const {
      const std::string varName = mode.getName() + "_dist_capacity";

      if (!mode.hasVar(varName)) {
        Expr posCapacity = Var::make(varName, Int());
        mode.addVar(varName, posCapacity);
        return posCapacity;
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
