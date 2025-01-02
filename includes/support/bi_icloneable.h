//
// Created by holynova on 2025/1/2.
//

#ifndef BATMANINFER_BI_ICLONEABLE_H
#define BATMANINFER_BI_ICLONEABLE_H

#include <memory>

namespace BatmanInfer {

namespace misc {

    /** Clonable Interface */
    template <class T>
    class BIICloneable
    {
    public:
        /** Default virtual desctructor */
        virtual ~BIICloneable() = default;
        /** Provide a clone of the current object of class T
         *
         * @return Clone object of class T
         */
        virtual std::unique_ptr<T> clone() const = 0;
    };

} // namespace misc

} // namespace BatmanInfer

#endif //BATMANINFER_BI_ICLONEABLE_H
