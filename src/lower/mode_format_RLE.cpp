//
// Created by Daniel Donenfeld on 1/24/21.
//

// TODO:
//  - Use better format for storing RLE lengths, maybe packbits except use type size for code
//    - We could use a byte for code and reinterpret it to mean type size elements instead of bytes
//      however this could cause issues indexing into vals array (due to taco not expecting this and
//      performance issues from unaligned accesses)
//  - Implement Append functionality
//  - Add support for merging

#include "taco/lower/mode_format_RLE.h"

#include "ir/ir_generators.h"
#include "taco/ir/simplify.h"
#include "taco/util/strings.h"
#include "taco/error.h"

#include <limits>

using namespace std;
using namespace taco::ir;

namespace taco {

    RLEModeFormat::RLEModeFormat() :
        RLEModeFormat(true, true) {
    }

    RLEModeFormat::RLEModeFormat(bool isFull, bool isUnique, Datatype rle_elem_type, bool includeComments, long long allocSize) :
            ModeFormatImpl("rle", isFull, true, isUnique, false, true, false,
                           true, false, false, false, true,
                           false, false, false, true, true),
            includeComments(includeComments), allocSize(allocSize), rle_elem_type(rle_elem_type) {
      taco_uassert(rle_elem_type.isUInt());
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
                std::make_shared<RLEModeFormat>(isFull, isUnique, rle_elem_type, includeComments);
        return ModeFormat(rleVariant);
    }

    vector<Expr> RLEModeFormat::getArrays(Expr tensor, int mode,
                                                 int level) const {
        std::string arraysName = util::toString(tensor) + std::to_string(level);
        return {GetProperty::make(tensor, TensorProperty::Indices,
                                  level - 1, 0, arraysName + "_pos"),
                GetProperty::make(tensor, TensorProperty::Indices,
                                  level - 1, 1, arraysName + "_rle",
                                  rle_elem_type)};
    }

    Datatype RLEModeFormat::IndexArrayType(int i) const {
      switch (i){
        case 0: return Int();
        case 1 : return rle_elem_type;
        default: taco_ierror; return Int();
      }
    }

    Expr RLEModeFormat::getPosArray(ModePack pack) const {
        return pack.getArray(0);
    }

    Expr RLEModeFormat::getRleArray(ModePack pack) const {
      return pack.getArray(1);
    }

    Expr RLEModeFormat::getValsArray(Mode mode) const {
        return GetProperty::make(mode.getTensorExpr(), TensorProperty::Values);
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

    Expr RLEModeFormat::getRleCapacity(Mode mode) const {
      const std::string varName = mode.getName() + "_rle_size";

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
             (taco_ierror, 0);
              // getSizeArray(mode.getModePack());
    }

    bool RLEModeFormat::equals(const ModeFormatImpl& other) const {
        // TODO: Is this right?
        return ModeFormatImpl::equals(other) &&
               (dynamic_cast<const RLEModeFormat&>(other).allocSize == allocSize) &&
                (dynamic_cast<const RLEModeFormat&>(other).rle_elem_type == rle_elem_type) &&
                (dynamic_cast<const RLEModeFormat&>(other).includeComments == includeComments)
                ;
    }

    ModeFunction RLEModeFormat::coordIterBounds(std::vector<ir::Expr> parentCoords, Mode mode) const {
      ir::Expr coord = parentCoords.empty() ? 0 : parentCoords.back();
      Stmt indexDecl = VarDecl::make(getIndexVar(mode),Load::make(getPosArray(mode.getModePack()), coord));
      Stmt curLenDecl = VarDecl::make(getCurLenVar(mode),Load::make(getRleArray(mode.getModePack()), getIndexVar(mode)));
      return ModeFunction(Block::make(indexDecl, curLenDecl), {0, getWidth(mode)});
    }

    ModeFunction RLEModeFormat::coordIterAccess(ir::Expr parentPos, std::vector<ir::Expr> coords, Mode mode) const {
        // TODO: This assumes that we always access the next coordinate, I think that will always be fine
        //   as this level format stores values for every coordinate
        std::vector<Stmt> stmts{};

        // We need to decrement the current length
        stmts.push_back(Assign::make(getCurLenVar(mode), ir::Sub::make(getCurLenVar(mode), 1)));

        // We need a variable to store the index we use to access
        Expr pVar = Var::make("result_index" + mode.getName(), Int());
        stmts.push_back(VarDecl::make(pVar, getIndexVar(mode)));

        // If the current length is zero we need to update the index and length vars
        Stmt thenBlock = Block::make(
            Assign::make(getIndexVar(mode), ir::Add::make(getIndexVar(mode), 1)),
            Assign::make(getCurLenVar(mode),Load::make(getRleArray(mode.getModePack()),getIndexVar(mode)))
                                    );
        Stmt ifStmt = IfThenElse::make({Eq::make(getCurLenVar(mode), 0)},thenBlock);
        stmts.push_back(ifStmt);

        return ModeFunction(Block::make(stmts), {pVar, true});
    }

    ir::Expr RLEModeFormat::getCurLenVar(Mode mode) const {
        const std::string curLength = mode.getName() + "_cur_run_length";

        if (!mode.hasVar(curLength)) {
            Expr var = Var::make(curLength, rle_elem_type);
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

    ir::Expr RLEModeFormat::getSavePosVar(Mode mode) const {
      const std::string indexVar = mode.getName() + "_save_pos";

      if (!mode.hasVar(indexVar)) {
        Expr var = Var::make(indexVar, Int());
        mode.addVar(indexVar, var);
        return var;
      }

      return mode.getVar(indexVar);
    }

    ir::Expr RLEModeFormat::getPosOffsetVar(Mode mode) const {
      const std::string indexVar = mode.getName() + "_pos_off";

      if (!mode.hasVar(indexVar)) {
        Expr var = Var::make(indexVar, Int());
        mode.addVar(indexVar, var);
        return var;
      }

      return mode.getVar(indexVar);
    }


    ModeFunction RLEModeFormat::repeatIterBounds(ir::Expr parentPos, Mode mode) const{
      Expr pbegin = Load::make(getPosArray(mode.getModePack()), parentPos);
      Expr pend = Load::make(getPosArray(mode.getModePack()),
                             ir::Add::make(parentPos, 1));
      return ModeFunction(Stmt(), {pbegin, pend});
    }

    ModeFunction RLEModeFormat::repeatIterAccess(ir::Expr pos,
                                                  std::vector<ir::Expr> coords,
                                                  Mode mode) const{
      Expr rleArray = getRleArray(mode.getModePack());
      Expr stride = (int)mode.getModePack().getNumModes();
      Expr count = Load::make(rleArray, ir::Mul::make(pos, stride));
      return ModeFunction(Stmt(), {pos, 1, count, true});
    }

    ModeFunction RLEModeFormat::coordBounds(ir::Expr parentPos, Mode mode) const {
      Stmt comment = includeComments ? Comment::make("Call to RLEModeFormat::coordBounds!") : Stmt();
      return ModeFunction(comment, {0, getWidth(mode)});
    }

    ir::Stmt RLEModeFormat::getAppendCoord(ir::Expr pos, ir::Expr coord, Mode mode) const {
      taco_iassert(mode.getPackLocation() == 0);

//      Stmt c0 = includeComments ? Comment::make("--Call to RLEModeFormat::getAppendCoord!   --") : Stmt();
//      Stmt c1 = includeComments ? Comment::make("--End call to RLEModeFormat::getAppendCoord--") : Stmt();
//
      // if (coord>0 && vals[pos*stride] == vals[(pos-1)*stride]) { rle[(pos-1)*stride]++; pos--; } else { rle[pos] = 1; }
      // TODO: if used for an outer dimension, this check is wrong + insufficient (it only checks one value back instead of all of them)

//      Expr valsArray = getValsArray(mode);
//      Expr rleArray = getRleArray(mode.getModePack());
//      Expr stride = (int)mode.getModePack().getNumModes();
//
//      Expr posMul = ir::Mul::make(pos, stride);
//      Expr prevPosMul = ir::Mul::make(ir::Sub::make(pos,1), stride);
//      Expr coordGtZero = Gt::make(coord, 0);
//      Expr valsEq = Eq::make(Load::make(valsArray, posMul), Load::make(valsArray,prevPosMul));
//      Expr ifCond = And::make(coordGtZero, valsEq);
//
//      Stmt thenBlock = Block::make(
//              incrementRle(pos, -1, mode),
//              Assign::make(pos, ir::Sub::make(pos, 1))
//              );

      Stmt storeIdx = storeIntoRle(pos, 0, 1, Expr(), mode);
//      if(false) {
//        return IfThenElse::make(ifCond, thenBlock, storeIdx);
//      } else
        return storeIdx;
    }

    ir::Stmt RLEModeFormat::getAppendRepeat(ir::Expr pos, ir::Expr off, ir::Expr coord, ir::Expr repeat, ir::Expr valsCapacity, Mode mode) const {
      taco_iassert(mode.getPackLocation() == 0);

      Expr rleArray = getRleArray(mode.getModePack());
      Expr stride = (int)mode.getModePack().getNumModes();

      Expr rle_pos = ir::Mul::make(ir::Add::make(pos,off), stride);
      Expr repeat_sum = ir::Add::make(repeat, ir::Load::make(rleArray, rle_pos));
      Stmt storeIdx = storeIntoRle(pos, off, repeat_sum, valsCapacity, mode);
      return storeIdx;
    }


    ir::Stmt RLEModeFormat::getAppendEdges(ir::Expr parentPos, ir::Expr posBegin, ir::Expr posEnd, Mode mode) const {
      Stmt c0 = includeComments ? Comment::make("-- Call to RLEModeFormat::getAppendEdges!    --") : Stmt();
      Stmt c1 = includeComments ? Comment::make("-- End call to RLEModeFormat::getAppendEdges --") : Stmt();


      Expr posArray = getPosArray(mode.getModePack());
      ModeFormat parentModeType = mode.getParentModeType();
      Expr edges = (!parentModeType.defined() || parentModeType.hasAppend())
                   ? posEnd : ir::Sub::make(posEnd, posBegin);
      Stmt store = Store::make(posArray, ir::Add::make(parentPos, 1), edges);

      return Block::make(c0,store,c1);
    }

    ir::Expr RLEModeFormat::getSize(ir::Expr parentSize, Mode mode) const {
      return Load::make(getPosArray(mode.getModePack()), parentSize); // TODO: I don't think this is right
    }

    ir::Stmt RLEModeFormat::getAppendInitEdges(ir::Expr parentPosBegin, ir::Expr parentPosEnd, Mode mode) const {

      if (isa<ir::Literal>(parentPosBegin)) {
        taco_iassert(to<ir::Literal>(parentPosBegin)->equalsScalar(0));
        if(includeComments) {
          Stmt c0 = Comment::make("-- Call to RLEModeFormat::getAppendInitEdges (literal case)!    --");
          Stmt c1 = Comment::make("-- End call to RLEModeFormat::getAppendInitEdges (literal case) --");
          return Block::make(c0, c1);
        } else {
          return Stmt();
        }
      }

      Expr posArray = getPosArray(mode.getModePack());
      Expr posCapacity = getPosCapacity(mode);
      ModeFormat parentModeType = mode.getParentModeType();
      if (!parentModeType.defined() || parentModeType.hasAppend()) {
        Stmt c0 = includeComments ? Comment::make("-- Call to RLEModeFormat::getAppendInitEdges (parent append)!    --") : Stmt();
        Stmt c1 = includeComments ? Comment::make("-- End call to RLEModeFormat::getAppendInitEdges (parent append) --") : Stmt();

        return Block::make(c0,doubleSizeIfFull(posArray, posCapacity, parentPosEnd),c1);
      }

      Stmt c0 = includeComments ? Comment::make("-- Call to RLEModeFormat::getAppendInitEdges!    --") : Stmt();
      Stmt c1 = includeComments ? Comment::make("-- End call to RLEModeFormat::getAppendInitEdges --") : Stmt();

      Expr pVar = Var::make("p" + mode.getName(), Int());
      Expr lb = ir::Add::make(parentPosBegin, 1);
      Expr ub = ir::Add::make(parentPosEnd, 1);
      Stmt initPos = For::make(pVar, lb, ub, 1, Store::make(posArray, pVar, 0));
      Stmt maybeResizePos = atLeastDoubleSizeIfFull(posArray, posCapacity, parentPosEnd);
      return Block::make({c0, maybeResizePos, initPos, c1});
    }

    ir::Stmt RLEModeFormat::getAppendInitLevel(ir::Expr parentSize, ir::Expr size, Mode mode) const {
      Stmt c0 = includeComments ? Comment::make("-- Call to RLEModeFormat::getAppendInitLevel!    --") : Stmt();
      Stmt c1 = includeComments ? Comment::make("-- End call to RLEModeFormat::getAppendInitLevel --") : Stmt();

      const bool szPrevIsZero = isa<ir::Literal>(parentSize) &&
                                to<ir::Literal>(parentSize)->equalsScalar(0);

      Expr defaultCapacity = ir::Literal::make(allocSize, Datatype::Int32);
      Expr posArray = getPosArray(mode.getModePack());
      Expr initCapacity = szPrevIsZero ? defaultCapacity : ir::Add::make(parentSize, 1);
      Expr posCapacity = initCapacity;

      std::vector<Stmt> initStmts;
      initStmts.push_back(c0);
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

      if (mode.getPackLocation() == (mode.getModePack().getNumModes() - 1)) {
        Expr rleCapacity = getRleCapacity(mode);
        Expr rleArray = getRleArray(mode.getModePack());
        initStmts.push_back(VarDecl::make(rleCapacity, defaultCapacity));
        initStmts.push_back(Allocate::make(rleArray, rleCapacity));
//        auto forVar = Var::make("fill_i", Int32);
//        auto forLoop = For::make(forVar, 0, rleCapacity, 1,
//                                 Block::make(Store::make(rleArray, forVar, 1, false, ParallelUnit::DefaultUnit)));
//        initStmts.push_back(forLoop);
      }

      initStmts.push_back(c1);
      return Block::make(initStmts);
    }

    ir::Stmt RLEModeFormat::getAppendFinalizeLevel(ir::Expr parentSize, ir::Expr size, Mode mode) const {
      ModeFormat parentModeType = mode.getParentModeType();
      if ((isa<ir::Literal>(parentSize) && to<ir::Literal>(parentSize)->equalsScalar(1)) ||
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
      Stmt finalizeLoop = For::make(pVar, 1, ir::Add::make(parentSize, 1), 1, body);

      // Do compression
//      vector<Stmt> stmts;
//      Expr tnpos = Var::make("tnpos", Int32);
//      Expr tnval = Var::make("tnval", mode.getTensorExpr().type());
//      Expr vals = getValsArray(mode);
//
//      stmts.push_back(VarDecl::make(tnpos, 0));
//      stmts.push_back(VarDecl::make(tnval, Load::make(vals, tnpos)));
//
//      Expr itpos = Var::make("itpos", Int32);
//
//      For::make(itpos, 1, Load::make)
//      int32_t tnpos = 0;
//      double tnval = t_vals[0];
//
//      for (int32_t itpos = 1; itpos < t_pos[1]; itpos++){
//        double val = t->vals[itpos];
//        if (val == tnval){
//          t_rle[tnpos]++; // HACK: This assumes that this will never overflow (which should be fine for this manual test)
//        } else {
//          tnpos++;
//          tnval = t_vals[tnpos];
//        }
//      }
//
//      void* res = realloc(t->vals, sizeof(double) * (tnpos+1));
//      if(!res){
//        taco_uerror;
//      }
//      t->vals = (uint8_t*)res;


      Stmt c0 = includeComments ? Comment::make("-- Call to RLEModeFormat::getAppendFinalizeLevel!    --") : Stmt();
      Stmt c1 = includeComments ? Comment::make("-- End call to RLEModeFormat::getAppendFinalizeLevel --") : Stmt();

      return Block::make({c0, initCs, finalizeLoop, c1});
    }

    uint64_t RLEModeFormat::max_value() const {
      taco_iassert(rle_elem_type.isUInt());
      if (rle_elem_type == UInt8){
        return std::numeric_limits<uint8_t>::max();
      } else if (rle_elem_type == UInt16){
        return std::numeric_limits<uint16_t>::max();
      } else if (rle_elem_type == UInt32){
        return std::numeric_limits<uint32_t>::max();
      } else if (rle_elem_type == UInt64){
        return std::numeric_limits<uint64_t>::max();
      } else {
        taco_ierror;
        return 0;
      }
    }

    Stmt RLEModeFormat::storeIntoRle(ir::Expr posVar, ir::Expr off, ir::Expr repeat, ir::Expr valsCapacity, Mode mode) const {
      Expr rleArray = getRleArray(mode.getModePack());
      Expr valsArray = getValsArray(mode);
      Expr stride = (int)mode.getModePack().getNumModes();
      Expr rle_pos = ir::Mul::make(ir::Add::make(posVar,off), stride);

      auto p = [&](Expr off2 = 0){
          return ir::Mul::make(ir::Add::make(posVar,off2), stride);
      };

      auto f = [&](Expr val, Expr off2 = 0){
        return Store::make(rleArray, p(ir::Add::make(off, off2)), val);
      };

      uint64_t maxRle = max_value();
      if(const ir::Literal* lit = repeat.as<ir::Literal>()){
        taco_iassert(lit->type.isIntegral());

        if(lit->getIntValue() == 1){
          return Block::make(doubleSizeIfFull(rleArray, getRleCapacity(mode), p(off)),
                             f(1));
        }

        int64_t repeatLit = lit->getIntValue();
        int64_t loopNum = repeatLit / maxRle;
        int64_t left_over = repeatLit % maxRle;

        vector<Stmt> stmts;
        if(repeatLit < maxRle){
          stmts.push_back(doubleSizeIfFull(rleArray, getRleCapacity(mode), p(off)));
          stmts.push_back(f(repeatLit, 0));
        } else {
          taco_iassert(loopNum > 0); // repeatLit is greater than the current RLE slot, so we can assume the loop
                                        // runs more than once.
          stmts.push_back(doubleSizeIfFull(valsArray, valsCapacity, ir::Add::make(posVar, loopNum)));
          stmts.push_back(doubleSizeIfFull(rleArray, getRleCapacity(mode), ir::Add::make(posVar, loopNum)));
          if (left_over == 0) {
            loopNum -= 1;
            stmts.push_back(f(maxRle, 0));
          } else{
            stmts.push_back(f(left_over, 0));
          }

          for(int64_t i=0; i < loopNum; i++){
            stmts.push_back(f(maxRle,i+1));
            stmts.push_back(Store::make(valsArray, ir::Add::make(rle_pos, i+1), Load::make(valsArray, rle_pos)));
          }

          stmts.push_back(Assign::make(posVar, ir::Add::make(posVar,loopNum)));
        }
        return Block::make(stmts);
      } else {
        // We need to generate the insertion loop from above
        Expr loopVar = Var::make(mode.getName() + "_rle_store", UInt64);
        Expr loopBound = Var::make(mode.getName() + "_rle_store_bound", UInt64);
        Expr rleLeftOver = Var::make(mode.getName() + "_rle_left_over", UInt64);
        Expr loopStart = Var::make(mode.getName() + "_rle_loop_start", UInt64);
        Expr repeatVar = Var::make(mode.getName() + "_rle_repeat", UInt64);


        vector<Stmt> stmts;
        stmts.push_back(VarDecl::make(repeatVar, repeat));
        stmts.push_back(VarDecl::make(loopBound, ir::Div::make(repeatVar, maxRle)));
        Expr remExpr = ir::Rem::make(repeatVar, maxRle);
        stmts.push_back(VarDecl::make(rleLeftOver, remExpr));
        stmts.push_back(VarDecl::make(loopStart, 0));

        stmts.push_back(doubleSizeIfFull(rleArray, getRleCapacity(mode), ir::Add::make(posVar,loopBound)));
        stmts.push_back(doubleSizeIfFull(valsArray, valsCapacity, ir::Add::make(posVar,loopBound)));
        stmts.push_back(IfThenElse::make(Eq::make(rleLeftOver, 0),
                                         Block::make(Assign::make(loopStart, 1),
                                                     f(maxRle))));
        stmts.push_back(For::make(loopVar, loopStart, loopBound, 1,
                                  Block::make(f(maxRle, ir::Add::make(loopVar,1)),
                                              Store::make(valsArray, p(ir::Add::make(ir::Add::make(off, loopVar),1)), Load::make(valsArray, rle_pos)))));
        stmts.push_back(IfThenElse::make(Gt::make(remExpr, 0),
                                         f(remExpr)));

        stmts.push_back(Assign::make(posVar, ir::Add::make(posVar,loopBound)));
        auto ret = Block::make(stmts);
        return ret;
      }
    }

}