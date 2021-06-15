#ifndef TACO_MODE_FORMAT_LZ77_H
#define TACO_MODE_FORMAT_LZ77_H

#include "taco/lower/mode_format_impl.h"

namespace taco {
    class LZ77ModeFormat : public ModeFormatImpl {
    public:
        using ModeFormatImpl::getInsertCoord;

        LZ77ModeFormat();
        LZ77ModeFormat(bool isFull, bool isOrdered,
                                bool isUnique, bool isZeroless,
                                long long allocSize = DEFAULT_ALLOC_SIZE);

        ~LZ77ModeFormat() override = default;

        ModeFormat copy(std::vector<ModeFormat::Property> properties) const override;

        ModeFunction posIterBounds(ir::Expr parentPos, Mode mode) const override;
        ModeFunction posIterAccess(ir::Expr pos, std::vector<ir::Expr> coords,
                                   Mode mode) const override;

        ModeFunction coordBounds(ir::Expr parentPos, Mode mode) const override;

        std::vector<ir::Expr> getArrays(ir::Expr tensor, int mode,
                                        int level) const override;

        ir::Expr getWidth(Mode mode) const override;

        ModeFunction getFillRegion(ir::Expr pos, std::vector<ir::Expr> coords,
                                   Mode mode) const override;

    protected:
        ir::Expr getPosArray(ModePack pack) const;
        ir::Expr getDistArray(ModePack pack) const;
        ir::Expr getRunArray(ModePack pack) const;

        ir::Expr getCoordVar(Mode mode) const;
        ir::Expr getPosCoordVar(Mode mode) const;

        bool equals(const ModeFormatImpl& other) const override;

        const long long allocSize;
    };
}


#endif //TACO_MODE_FORMAT_LZ77_H
