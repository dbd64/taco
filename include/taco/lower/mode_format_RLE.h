//
// Created by Daniel Donenfeld on 1/24/21.
//

#ifndef TACO_MODE_FORMAT_RLE_H
#define TACO_MODE_FORMAT_RLE_H

#include "taco/lower/mode_format_impl.h"

namespace taco {

    class RLEModeFormat : public ModeFormatImpl {
    public:
        RLEModeFormat();
        RLEModeFormat(bool isFull, bool isUnique,
                      bool includeComments = false,
                      long long allocSize = DEFAULT_ALLOC_SIZE);

        ~RLEModeFormat() override {}

        bool includeComments;

        ModeFormat copy(std::vector<ModeFormat::Property> properties) const override;

        ir::Expr getWidth(Mode mode) const override;

        ModeFunction coordIterBounds(std::vector<ir::Expr> parentCoords,
                                     Mode mode) const override;

        ModeFunction coordIterAccess(ir::Expr parentPos,
                                     std::vector<ir::Expr> coords,
                                     Mode mode) const override;


        ModeFunction repeatIterBounds(ir::Expr parentPos, Mode mode) const override;
        ModeFunction repeatIterAccess(ir::Expr pos, std::vector<ir::Expr> coords, Mode mode) const override;

        ModeFunction coordBounds(ir::Expr parentPos, Mode mode) const override;

        ir::Stmt getAppendCoord(ir::Expr pos, ir::Expr coord,
                                Mode mode) const override;
        ir::Stmt getAppendEdges(ir::Expr parentPos, ir::Expr posBegin,
                                ir::Expr posEnd, Mode mode) const override;
        ir::Expr getSize(ir::Expr parentSize, Mode mode) const override;
        ir::Stmt getAppendInitEdges(ir::Expr parentPosBegin,
                                    ir::Expr parentPosEnd, Mode mode) const override;
        ir::Stmt getAppendInitLevel(ir::Expr parentSize, ir::Expr size,
                                    Mode mode) const override;
        ir::Stmt getAppendFinalizeLevel(ir::Expr parentSize, ir::Expr size,
                                        Mode mode) const override;

        std::vector<ir::Expr> getArrays(ir::Expr tensor, int mode,
                                        int level) const override;

        ir::Expr getSavePosVar(Mode mode) const;
        ir::Expr getPosOffsetVar(Mode mode) const;

    protected:
        // We use a pos array to distinguish different rows/cols/modes,
        ir::Expr getPosArray(ModePack pack) const;
        ir::Expr getRleArray(ModePack pack) const;

        ir::Expr getValsArray(Mode mode) const;

        // Used for appending
        ir::Expr getPosCapacity(Mode mode) const;
        ir::Expr getRleCapacity(Mode mode) const;

        // Variables used to maintain state needed for iteration
        ir::Expr getCurLenVar(Mode mode) const;
        ir::Expr getIndexVar(Mode mode) const;

        bool equals(const ModeFormatImpl& other) const override;

        const long long allocSize;

    };

}

#endif //TACO_MODE_FORMAT_RLE_H
