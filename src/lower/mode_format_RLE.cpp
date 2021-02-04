//
// Created by Daniel Donenfeld on 1/24/21.
//

// I THINK WE NEED TO DO COORDINATE ITERATION
// Each position in the vals array may correspond
// to more than one coordinate. If we instead iterate
// over the coordinates we can return the same
// position multiple times.
// TODO: How can I maintain state iteration?
//   - I think I can initialize state in the coord_bounds function, but I need to test this
//   - We should only need the pos array, the data can be stored entirely in the vals array.
//       - TODO: I will need to see how this interacts with other level formats though (and itself)

#include "../../include/taco/lower/mode_format_RLE.h"

#include "ir/ir_generators.h"
#include "taco/ir/simplify.h"
#include "taco/util/strings.h"
#include "taco/error.h"

using namespace std;
using namespace taco::ir;

namespace taco {

    RLEModeFormat::RLEModeFormat() :
        RLEModeFormat(false, true) {
    }

    RLEModeFormat::RLEModeFormat(bool isFull, bool isUnique, long long allocSize) :
            ModeFormatImpl("rle", isFull, true, isUnique, false, true,
                           true, false, false, false, true),
            allocSize(allocSize) {
    }

    ModeFormat RLEModeFormat::copy(
            vector<ModeFormat::Property> properties) const {
        bool isFull = this->isFull;
        bool isUnique = this->isUnique;
        for (const auto property : properties) {
            switch (property) {
                case ModeFormat::FULL:
                    isFull = true;
                    break;
                case ModeFormat::NOT_FULL:
                    isFull = false;
                    break;
                case ModeFormat::UNIQUE:
                    isUnique = true;
                    break;
                case ModeFormat::NOT_UNIQUE:
                    isUnique = false;
                    break;
                default:
                    break;
            }
        }
        const auto rleVariant =
                std::make_shared<RLEModeFormat>(isFull, isUnique);
        return ModeFormat(rleVariant);
    }

    vector<Expr> RLEModeFormat::getArrays(Expr tensor, int mode,
                                                 int level) const {
        std::string arraysName = util::toString(tensor) + std::to_string(level);
        return {GetProperty::make(tensor, TensorProperty::Dimension, mode),
                GetProperty::make(tensor, TensorProperty::Indices,
                                  level - 1, 0, arraysName + "_pos"),
                GetProperty::make(tensor, TensorProperty::Indices,
                                  level - 1, 0, arraysName + "_rle")};
    }

    Expr RLEModeFormat::getSizeArray(ModePack pack) const {
        return pack.getArray(0);
    }

    Expr RLEModeFormat::getPosArray(ModePack pack) const {
        return pack.getArray(1);
    }

    Expr RLEModeFormat::getRleArray(ModePack pack) const {
        return pack.getArray(2);
    }

    Expr RLEModeFormat::getPosCapacity(Mode mode) const {
        const std::string varName = mode.getName() + "_pos_size";

        if (!mode.hasVar(varName)) {
            Expr posCapacity = Var::make(varName, Int());
            mode.addVar(varName, posCapacity);
            return posCapacity;
        }

        return mode.getVar(varName);
    }

    Expr RLEModeFormat::getWidth(Mode mode) const {
        return (mode.getSize().isFixed() && mode.getSize().getSize() < 16) ?
               (int)mode.getSize().getSize() :
               getSizeArray(mode.getModePack());
    }

    bool RLEModeFormat::equals(const ModeFormatImpl& other) const {
        // TODO: Is this right?
        return ModeFormatImpl::equals(other) &&
               (dynamic_cast<const RLEModeFormat&>(other).allocSize == allocSize);
    }

    ModeFunction RLEModeFormat::coordIterBounds(std::vector<ir::Expr> parentCoords, Mode mode) const {
        Stmt curLenDecl = VarDecl::make(getCurLenVar(mode),0);
        Stmt indexDecl = VarDecl::make(getIndexVar(mode),0);
        return ModeFunction(Block::make(curLenDecl, indexDecl), {0, getWidth(mode)});
    }

    ModeFunction RLEModeFormat::coordIterAccess(ir::Expr parentPos, std::vector<ir::Expr> coords, Mode mode) const {
        std::vector<Stmt> stmts{};

        // We need to decrement the current length
        stmts.push_back(Assign::make(getCurLenVar(mode), Sub::make(getCurLenVar(mode), 1)));

        // We need a variable to store the index we use to access
        Expr pVar = Var::make("result_index" + mode.getName(), Int());
        stmts.push_back(VarDecl::make(pVar, getIndexVar(mode)));

        // If the current length is zero we need to update the index and length vars
        Stmt thenBlock = Block::make(
            Assign::make(getIndexVar(mode), Add::make(getIndexVar(mode), 1)),
            Assign::make(getCurLenVar(mode),Load::make(getRleArray(mode.getModePack()),getIndexVar(mode)))
                                    );
        Stmt ifStmt = IfThenElse::make({Eq::make(getCurLenVar(mode), 0)},thenBlock);

        return ModeFunction(Block::make(stmts), {pVar, true});
    }

    ir::Expr RLEModeFormat::getCurLenVar(Mode mode) const {
        const std::string curLength = mode.getName() + "_cur_run_length";

        if (!mode.hasVar(curLength)) {
            Expr var = Var::make(curLength, Int());
            mode.addVar(curLength, var);
            return var;
        }

        return mode.getVar(curLength);
    }

    ir::Expr RLEModeFormat::getIndexVar(Mode mode) const {
        const std::string indexVar = mode.getName() + "_cur_index";

        if (!mode.hasVar(indexVar)) {
            Expr var = Var::make(indexVar, Int());
            mode.addVar(indexVar, var);
            return var;
        }

        return mode.getVar(indexVar);
    }

}