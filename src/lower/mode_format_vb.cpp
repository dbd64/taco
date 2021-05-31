#include "taco/lower/mode_format_vb.h"

#include "ir/ir_generators.h"
#include "taco/ir/simplify.h"
#include "taco/util/strings.h"

using namespace std;
using namespace taco::ir;

namespace taco {

    VariableBlockModeFormat::VariableBlockModeFormat() :
            VariableBlockModeFormat(false, true, true, false, false) {
    }

    VariableBlockModeFormat::VariableBlockModeFormat(bool isFull, bool isOrdered,
                                               bool isUnique, bool isZeroless,
                                               bool isLastValueFill,
                                               long long allocSize) :
            ModeFormatImpl("variableblock", isFull, isOrdered, isUnique, false, true,
                           isZeroless, false, true, false, false,
                           true, false, false,
                           false, false, isLastValueFill),
            allocSize(allocSize) {
    }

    ModeFormat VariableBlockModeFormat::copy(
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
              std::make_shared<VariableBlockModeFormat>(isFull, isOrdered, isUnique,
                                                     isZeroless);
      return ModeFormat(compressedVariant);
    }

    ModeFunction VariableBlockModeFormat::posIterBounds(Expr parentPos,
                                                     Mode mode) const {
//      Expr curr_crd = getCurrentCoord(mode);
//      Expr curr_crd_idx = getCurrentCoordIdx(mode);

      Expr pbegin = Load::make(getSizePosArray(mode.getModePack()), parentPos);
      Expr pend = Load::make(getSizePosArray(mode.getModePack()),
                             ir::Add::make(parentPos, 1));

//      Stmt curr_crd_init = VarDecl::make(curr_crd, 0);
//      Expr pbeginSz = Load::make(getSizePosArray(mode.getModePack()), parentPos);
//      Stmt curr_crd_idx_init = VarDecl::make(curr_crd_idx, pbeginSz);
//      Stmt blk = Block::make(curr_crd_init, curr_crd_idx_init);

      Expr startPos = getStartPos(mode);
      Stmt blk =  VarDecl::make(startPos, pbegin);

      return ModeFunction(blk, {pbegin, pend});
    }

    ModeFunction VariableBlockModeFormat::coordBounds(Expr parentPos,
                                                   Mode mode) const {
      taco_not_supported_yet;
//      Expr pend = Load::make(getPosArray(mode.getModePack()),
//                             ir::Add::make(parentPos, 1));
//      Expr coordend = Load::make(getCoordArray(mode.getModePack()), ir::Sub::make(pend, 1));
//      Expr sizeend = Load::make(getSizeArray(mode.getModePack()), ir::Sub::make(pend, 1));
//      return ModeFunction(Stmt(), {0, ir::Add::make(coordend, sizeend)});
    }



    ModeFunction VariableBlockModeFormat::posIterAccess(ir::Expr pos,
                                                     std::vector<ir::Expr> coords,
                                                     Mode mode) const {
      taco_iassert(mode.getPackLocation() == 0);

      Expr coordOffset = ir::Sub::make(pos, getStartPos(mode));
      Expr parentCoord = coords.size() > 1 ? coords[coords.size() - 2] : coords[coords.size() - 1];
      for (auto& coord : coords) {
        std::cout << coord << ", ";
      }
      std::cout << std::endl;
      return ModeFunction(Stmt(), {ir::Add::make(parentCoord, coordOffset), true});

//      Expr idxArray = getCoordArray(mode.getModePack());
//      Expr szArray = getSizeArray(mode.getModePack());
//
//      Expr curr_crd = getCurrentCoord(mode);
//      Expr curr_crd_idx = getCurrentCoordIdx(mode);
//
//      // if (curr_crd == size[i]) { curr_crd=0;curr_crd_idx++; }
//      // idx = crd[curr_crd_idx] + curr_crd;
//      // curr_crd++;
//
//      Expr stride = (int)mode.getModePack().getNumModes();
//      Expr idx = Load::make(idxArray, ir::Mul::make(curr_crd_idx, stride));
//      Expr size = Load::make(szArray, ir::Mul::make(curr_crd_idx, stride));
//      Expr retIdx = Var::make("idx", Int());
//
//      Stmt ifStmt = IfThenElse::make(Eq::make(curr_crd, size),
//                               Block::make(Assign::make(curr_crd, 0),
//                                           addAssign(curr_crd_idx, 1)));
//
//      Stmt blk = Block::make(ifStmt,
//                             VarDecl::make(retIdx, ir::Add::make(idx, curr_crd)),
//                             addAssign(curr_crd, 1));
//
//
//      return ModeFunction(blk, {retIdx, true});
    }

    vector<Expr> VariableBlockModeFormat::getArrays(Expr tensor, int mode,
                                                 int level) const {
      std::string arraysName = util::toString(tensor) + std::to_string(level);
      return {GetProperty::make(tensor, TensorProperty::Indices,
                                level - 1, 0, arraysName + "_pos")};

}

    Expr VariableBlockModeFormat::getSizePosArray(ModePack pack) const {
      return pack.getArray(0);
    }

    Expr VariableBlockModeFormat::getStartPos(Mode mode) const {
      const std::string varName = mode.getName() + "_start_pos";

      if (!mode.hasVar(varName)) {
        Expr idxCapacity = Var::make(varName, Int());
        mode.addVar(varName, idxCapacity);
        return idxCapacity;
      }

      return mode.getVar(varName);
    }

    Expr VariableBlockModeFormat::getWidth(Mode mode) const {
      return ir::Literal::make(allocSize, Datatype::Int32);
    }

    bool VariableBlockModeFormat::equals(const ModeFormatImpl& other) const {
      return ModeFormatImpl::equals(other) &&
             (dynamic_cast<const VariableBlockModeFormat&>(other).allocSize == allocSize);
    }

}
