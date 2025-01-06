//
// Created by Mason on 2024/12/26.
//

#ifndef BATMANINFER_BI_TYPES_HPP
#define BATMANINFER_BI_TYPES_HPP

/** The following symbols have been moved to:
 * half
 * PermutationVector
 * Format
 * DataType
 * DataLayout
 * DataLayoutDimension
 * PadStrideInfo
 * WeightFormat
 * Channel
 * DimensionRoundingType
 */
#include <data/core/core_types.hpp>

#include <data/core/bi_coordinates.hpp>

#include <data/bi_tensor_shape.hpp>

#include <data/core/bi_size_2D.h>

#include <data/core/bi_size_3D.h>

#include <function_info/bi_activationLayerInfo.h>

#include <string>
#include <utility>

namespace BatmanInfer {

    /** Bidirectional strides */
    using BiStrides = BICoordinates;

    /** Padding information as a pair of unsigned int start/end */
    using PaddingInfo = std::pair<uint32_t, uint32_t>;

    /** List of padding information */
    using PaddingList = std::vector<PaddingInfo>;

    /** Coordinate type */
    struct BICoordinates2D
    {
        int32_t x; /**< X coordinates */
        int32_t y; /**< Y coordinates */
    };

    /** Container for 2D border size */
    struct BIBorderSize {
        /** Empty border, i.e. no border */
        constexpr BIBorderSize() noexcept: top{0}, right{0}, bottom{0}, left{0} {
        }

        /** Border with equal size around the 2D plane */
        explicit constexpr BIBorderSize(unsigned int size) noexcept: top{size}, right{size}, bottom{size}, left{size} {
        }

        /** Border with same size for top/bottom and left/right */
        constexpr BIBorderSize(unsigned int top_bottom, unsigned int left_right)
                : top{top_bottom}, right{left_right}, bottom{top_bottom}, left{left_right} {
        }

        /** Border with different sizes */
        constexpr BIBorderSize(unsigned int top, unsigned int right, unsigned int bottom, unsigned int left)
                : top{top}, right{right}, bottom{bottom}, left{left} {
        }

        /** Check if the entire border is zero */
        constexpr bool empty() const {
            return top == 0 && right == 0 && bottom == 0 && left == 0;
        }

        /** Check if the border is the same size on all sides */
        constexpr bool uniform() const {
            return top == right && top == bottom && top == left;
        }

        /** Scale this border size.
         *
         * @param[in] scale Scale to multiply border size by.
         *
         * @return *this.
         */
        BIBorderSize &operator*=(float scale) {
            top *= scale;
            right *= scale;
            bottom *= scale;
            left *= scale;

            return *this;
        }

        /** Scale a copy of this border size.
         *
         * @param[in] scale Scale to multiply border size by.
         *
         * @return a scaled copy of this.
         */
        BIBorderSize operator*(float scale) {
            BIBorderSize size = *this;
            size *= scale;

            return size;
        }

        /** Check equality with another BIBorderSize struct
         *
         * @param[in] rhs other struct to check against
         *
         * @return true if they are equal
         */
        bool operator==(const BIBorderSize &rhs) const {
            return (top == rhs.top) && (right == rhs.right) && (bottom == rhs.bottom) && (left == rhs.left);
        }

        /** Check non-equality with another BIBorderSize struct
         *
         * @param[in] rhs other struct to check against
         *
         * @return true if they are different
         */
        bool operator!=(const BIBorderSize &rhs) const {
            return !(*this == rhs);
        }

        /** Limit this border size.
         *
         * @param[in] limit Border size to limit this border size to.
         */
        void limit(const BIBorderSize &limit) {
            top    = std::min(top, limit.top);
            right  = std::min(right, limit.right);
            bottom = std::min(bottom, limit.bottom);
            left   = std::min(left, limit.left);
        }

        unsigned int top;    /**< top of the border */
        unsigned int right;  /**< right of the border */
        unsigned int bottom; /**< bottom of the border */
        unsigned int left;   /**< left of the border */
    };

    /**
     * @brief 用于二维内边距大小的容器
     */
    using BIPaddingSize = BIBorderSize;

    /**
     * @brief 窗口有效区域的容器
     */
    struct BIValidRegion {

        BIValidRegion() : anchor{}, shape{} {
        }

        BIValidRegion(const BIValidRegion &) = default;

        BIValidRegion(BIValidRegion &&) = default;

        BIValidRegion &operator=(const BIValidRegion &) = default;

        BIValidRegion &operator=(BIValidRegion &&) = default;

        ~BIValidRegion() = default;

        BIValidRegion(const BICoordinates &an_anchor,
                      const BITensorShape &a_shape) : anchor{an_anchor}, shape{a_shape} {
            anchor.set_num_dimensions(std::max(anchor.num_dimensions(), shape.num_dimensions()));
        }

        BIValidRegion(const BICoordinates &an_anchor,
                      const BITensorShape &a_shape,
                      size_t num_dimensions) : anchor{an_anchor}, shape{a_shape} {
            BI_COMPUTE_ERROR_ON(num_dimensions < std::max(anchor.num_dimensions(), shape.num_dimensions()));
            anchor.set_num_dimensions(num_dimensions);
        }

        /**
         * @brief 返回给定维度 @p d 的有效区域的起始位置。
         * @param d
         * @return
         */
        int start(unsigned int d) const {
            return anchor[d];
        }

        /**
         * @brief 返回给定维度 @p d 的有效区域的结束部分。
         * @param d
         * @return
         */
        int end(unsigned int d) const {
            return anchor[d] + shape[d];
        }

        /**
         * @brief 访问器用于设置一个维度的锚点和形状的值。
         * @param dimension 设置值的维度。
         * @param start 要在锚点中设置的维度值。
         * @param size
         * @return
         */
        BIValidRegion &set(size_t dimension,
                           int start,
                           size_t size) {
            anchor.set(dimension, start);
            shape.set(dimension, size);
            return *this;
        }

        /**
         * @brief Check whether two valid regions are equal
         * @param lhs
         * @param rhs
         * @return
         */
        inline friend bool operator==(const BIValidRegion &lhs, const BIValidRegion &rhs);

        /**
         * @brief 有效区域开始的锚点
         */
        BICoordinates anchor;

        /**
         * @brief 有效区域的形状
         */
        BITensorShape shape;
    };

    inline bool operator==(const BIValidRegion &lhs, const BIValidRegion &rhs) {
        return (lhs.anchor == rhs.anchor) && (lhs.shape == rhs.shape);
    }

    /**
     * @brief IO格式信息类
     */
    struct BIIOFormatInfo {
        /**
         * @brief 打印浮点数的精度类型
         */
        enum class PrecisionType {
            /**
             * @brief 默认精度为当前流所具有的精度
             */
            Default,
            /**
             * @brief 用户通过精度参数指定的自定义精度
             */
            Custom,
            /**
             * @brief 浮点表示的最大精度
             */
            Full
        };

        enum class PrintRegion {
            /**
             * @brief 打印 Tensor 对象的有效区域
             */
            ValidRegion,
            /**
             * @brief 打印没有填充的张量对象
             */
            NoPadding,
            /**
             * @brief 打印包含填充的张量对象
             */
            Full
        };

        explicit BIIOFormatInfo(PrintRegion print_region = PrintRegion::ValidRegion,
                                PrecisionType precision_type = PrecisionType::Default,
                                unsigned int precision = 10,
                                bool align_columns = true,
                                std::string element_delim = "",
                                std::string row_delim = "\n") : print_region(print_region),
                                                                precision_type(precision_type),
                                                                precision(precision),
                                                                element_delim(std::move(element_delim)),
                                                                row_delim(std::move(row_delim)),
                                                                align_columns(align_columns) {

        }

        /** Area to be printed by Tensor objects */
        PrintRegion   print_region;
        /** Floating point precision type */
        PrecisionType precision_type;
        /** Floating point precision */
        unsigned int  precision;
        /** Element delimeter */
        std::string   element_delim;
        /** Row delimeter */
        std::string   row_delim;
        /** Align columns */
        bool          align_columns;
    };

    /** Available Detection Output code types */
    enum class BIDetectionOutputLayerCodeType
    {
        CORNER,      /**< Use box corners */
        CENTER_SIZE, /**< Use box centers and size */
        CORNER_SIZE, /**< Use box centers and size */
        TF_CENTER    /**< Use box centers and size but flip x and y co-ordinates */
    };

    /** The normalization type used for the normalization layer */
    enum class BINormType
    {
        IN_MAP_1D, /**< Normalization applied within the same map in 1D region */
        IN_MAP_2D, /**< Normalization applied within the same map in 2D region */
        CROSS_MAP  /**< Normalization applied cross maps */
    };

    /** Available pooling types */
    enum class BIPoolingType
    {
        MAX, /**< Max Pooling */
        AVG, /**< Average Pooling */
        L2   /**< L2 Pooling */
    };

    /** Interpolation method */
    enum class BIInterpolationPolicy
    {
        NEAREST_NEIGHBOR, /**< Output values are defined to match the source pixel whose center is nearest to the sample position */
        BILINEAR,         /**< Output values are defined by bilinear interpolation between the pixels */
        AREA, /**< Output values are determined by averaging the source pixels whose areas fall under the area of the destination pixel, projected onto the source image */
    };

    /** Detection Output layer info */
    class BIDetectionOutputLayerInfo final
    {
    public:
        /** Default Constructor */
        BIDetectionOutputLayerInfo()
            : _num_classes(),
              _share_location(),
              _code_type(BIDetectionOutputLayerCodeType::CORNER),
              _keep_top_k(),
              _nms_threshold(),
              _top_k(),
              _background_label_id(),
              _confidence_threshold(),
              _variance_encoded_in_target(false),
              _eta(),
              _num_loc_classes()
        {
            _num_loc_classes = _share_location ? 1 : _num_classes;
        }
        /** Constructor
         *
         * @param[in] num_classes                Number of classes to be predicted.
         * @param[in] share_location             If true, bounding box are shared among different classes.
         * @param[in] code_type                  Type of coding method for bbox.
         * @param[in] keep_top_k                 Number of total bounding boxes to be kept per image after NMS step.
         * @param[in] nms_threshold              Threshold to be used in NMS.
         * @param[in] top_k                      (Optional) Number of boxes per image with top confidence scores that are fed into the NMS algorithm. Default set to -1.
         * @param[in] background_label_id        (Optional) Background label ID. If there is no background class, set it as -1.
         * @param[in] confidence_threshold       (Optional) Only consider detections whose confidences are larger than a threshold. Default set to -FLT_MAX.
         * @param[in] variance_encoded_in_target (Optional) If true, variance is encoded in target. Otherwise we need to adjust the predicted offset accordingly.Default set to false.
         * @param[in] eta                        (Optional) Eta.
         */
        BIDetectionOutputLayerInfo(int                            num_classes,
                                   bool                           share_location,
                                   BIDetectionOutputLayerCodeType code_type,
                                   int                            keep_top_k,
                                   float                          nms_threshold,
                                   int                            top_k                = -1,
                                   int                            background_label_id  = -1,
                                   float                          confidence_threshold = std::numeric_limits<float>::lowest(),
                                   bool                           variance_encoded_in_target = false,
                                   float                          eta                        = 1)
            : _num_classes(num_classes),
              _share_location(share_location),
              _code_type(code_type),
              _keep_top_k(keep_top_k),
              _nms_threshold(nms_threshold),
              _top_k(top_k),
              _background_label_id(background_label_id),
              _confidence_threshold(confidence_threshold),
              _variance_encoded_in_target(variance_encoded_in_target),
              _eta(eta),
              _num_loc_classes()
        {
            _num_loc_classes = _share_location ? 1 : _num_classes;
        }
        /** Get num classes. */
        int num_classes() const
        {
            return _num_classes;
        }
        /** Get share location. */
        bool share_location() const
        {
            return _share_location;
        }
        /** Get detection output code type. */
        BIDetectionOutputLayerCodeType code_type() const
        {
            return _code_type;
        }
        /** Get if variance encoded in target. */
        bool variance_encoded_in_target() const
        {
            return _variance_encoded_in_target;
        }
        /** Get the number of total bounding boxes to be kept per image. */
        int keep_top_k() const
        {
            return _keep_top_k;
        }
        /** Get nms threshold. */
        float nms_threshold() const
        {
            return _nms_threshold;
        }
        /** Get eta. */
        float eta() const
        {
            return _eta;
        }
        /** Get background label ID. */
        int background_label_id() const
        {
            return _background_label_id;
        }
        /** Get confidence threshold. */
        float confidence_threshold() const
        {
            return _confidence_threshold;
        }
        /** Get top K. */
        int top_k() const
        {
            return _top_k;
        }
        /** Get number of location classes. */
        int num_loc_classes() const
        {
            return _num_loc_classes;
        }

    private:
        int                            _num_classes;
        bool                           _share_location;
        BIDetectionOutputLayerCodeType _code_type;
        int                            _keep_top_k;
        float                          _nms_threshold;
        int                            _top_k;
        int                            _background_label_id;
        float                          _confidence_threshold;
        bool                           _variance_encoded_in_target;
        float                          _eta;
        int                            _num_loc_classes;
    };

    /** Bounding Box Transform information class */
    class BIBoundingBoxTransformInfo final
    {
    public:
        /** Constructor
         *
         * @param[in] img_width                Width of the original image
         * @param[in] img_height               Height, of the original image
         * @param[in] scale                    Scale of the original image
         * @param[in] apply_scale              (Optional)Re-apply scaling after transforming the boxes. Defaults to false
         * @param[in] weights                  (Optional)Weights [wx, wy, ww, wh] for the deltas. Defaults to all ones
         * @param[in] correct_transform_coords (Optional)Correct bounding box transform coordinates. Defaults to false
         * @param[in] bbox_xform_clip          (Optional)Minimum bounding box width and height after bounding box transformation in log-space. Defaults to log(1000/16)
         */
        BIBoundingBoxTransformInfo(float                      img_width,
                                   float                      img_height,
                                   float                      scale,
                                   bool                       apply_scale              = false,
                                   const std::array<float, 4> weights                  = {{1.f, 1.f, 1.f, 1.f}},
                                   bool                       correct_transform_coords = false,
                                   float                      bbox_xform_clip          = 4.135166556742356f)
            : _img_width(img_width),
              _img_height(img_height),
              _scale(scale),
              _apply_scale(apply_scale),
              _correct_transform_coords(correct_transform_coords),
              _weights(weights),
              _bbox_xform_clip(bbox_xform_clip)
        {
        }

        std::array<float, 4> weights() const
        {
            return _weights;
        }

        float bbox_xform_clip() const
        {
            return _bbox_xform_clip;
        }

        float img_height() const
        {
            return _img_height;
        }

        float img_width() const
        {
            return _img_width;
        }

        float scale() const
        {
            return _scale;
        }

        bool apply_scale() const
        {
            return _apply_scale;
        }

        bool correct_transform_coords() const
        {
            return _correct_transform_coords;
        }

    private:
        float                _img_width;
        float                _img_height;
        float                _scale;
        bool                 _apply_scale;
        bool                 _correct_transform_coords;
        std::array<float, 4> _weights;
        float                _bbox_xform_clip;
    };

    /** Available reduction operations */
    enum class BIReductionOperation
    {
        ARG_IDX_MAX, /**< Index of the max value */
        ARG_IDX_MIN, /**< Index of the min value */
        MEAN_SUM,    /**< Mean of sum */
        PROD,        /**< Product */
        SUM_SQUARE,  /**< Sum of squares */
        SUM,         /**< Sum */
        MIN,         /**< Min */
        MAX,         /**< Max */
    };

    /** Detection Output layer info */
    class BIDetectionPostProcessLayerInfo final
    {
    public:
        /** Default Constructor */
        BIDetectionPostProcessLayerInfo()
            : _max_detections(),
              _max_classes_per_detection(),
              _nms_score_threshold(),
              _iou_threshold(),
              _num_classes(),
              _scales_values(),
              _use_regular_nms(),
              _detection_per_class(),
              _dequantize_scores()
        {
        }
        /** Constructor
         *
         * @param[in] max_detections            Number of total detection.
         * @param[in] max_classes_per_detection Number of total classes to be kept after NMS step. Used in the Fast Non-Max-Suppression
         * @param[in] nms_score_threshold       Threshold to be used in NMS
         * @param[in] iou_threshold             Threshold to be used during the intersection over union.
         * @param[in] num_classes               Number of classes.
         * @param[in] scales_values             Scales values used for decode center size boxes.
         * @param[in] use_regular_nms           (Optional) Boolean to determinate if use regular or fast nms. Defaults to false.
         * @param[in] detection_per_class       (Optional) Number of detection per class. Used in the Regular Non-Max-Suppression. Defaults to 100.
         * @param[in] dequantize_scores         (Optional) If the scores need to be dequantized. Defaults to true.
         */
        BIDetectionPostProcessLayerInfo(unsigned int         max_detections,
                                        unsigned int         max_classes_per_detection,
                                        float                nms_score_threshold,
                                        float                iou_threshold,
                                        unsigned int         num_classes,
                                        std::array<float, 4> scales_values,
                                        bool                 use_regular_nms     = false,
                                        unsigned int         detection_per_class = 100,
                                        bool                 dequantize_scores   = true)
            : _max_detections(max_detections),
              _max_classes_per_detection(max_classes_per_detection),
              _nms_score_threshold(nms_score_threshold),
              _iou_threshold(iou_threshold),
              _num_classes(num_classes),
              _scales_values(scales_values),
              _use_regular_nms(use_regular_nms),
              _detection_per_class(detection_per_class),
              _dequantize_scores(dequantize_scores)
        {
        }
        /** Get max detections. */
        unsigned int max_detections() const
        {
            return _max_detections;
        }
        /** Get max_classes per detection. Used in the Fast Non-Max-Suppression.*/
        unsigned int max_classes_per_detection() const
        {
            return _max_classes_per_detection;
        }
        /** Get detection per class. Used in the Regular Non-Max-Suppression */
        unsigned int detection_per_class() const
        {
            return _detection_per_class;
        }
        /** Get nms threshold. */
        float nms_score_threshold() const
        {
            return _nms_score_threshold;
        }
        /** Get intersection over union threshold. */
        float iou_threshold() const
        {
            return _iou_threshold;
        }
        /** Get num classes. */
        unsigned int num_classes() const
        {
            return _num_classes;
        }
        /** Get if use regular nms. */
        bool use_regular_nms() const
        {
            return _use_regular_nms;
        }
        /** Get y scale value. */
        float scale_value_y() const
        {
            // Saved as [y,x,h,w]
            return _scales_values[0];
        }
        /** Get x scale value. */
        float scale_value_x() const
        {
            // Saved as [y,x,h,w]
            return _scales_values[1];
        }
        /** Get h scale value. */
        float scale_value_h() const
        {
            // Saved as [y,x,h,w]
            return _scales_values[2];
        }
        /** Get w scale value. */
        float scale_value_w() const
        {
            // Saved as [y,x,h,w]
            return _scales_values[3];
        }
        /** Get dequantize_scores value. */
        bool dequantize_scores() const
        {
            return _dequantize_scores;
        }

    private:
        unsigned int         _max_detections;
        unsigned int         _max_classes_per_detection;
        float                _nms_score_threshold;
        float                _iou_threshold;
        unsigned int         _num_classes;
        std::array<float, 4> _scales_values;
        bool                 _use_regular_nms;
        unsigned int         _detection_per_class;
        bool                 _dequantize_scores;
    };

    enum class BIConvertPolicy
	{
	    WRAP,    /**< Wrap around */
    	SATURATE /**< Saturate */
	};

    /** Generate Proposals Information class */
    class BIGenerateProposalsInfo
    {
    public:
        /** Constructor
         *
         * @param[in] im_width       Width of the original image
         * @param[in] im_height      Height of the original image
         * @param[in] im_scale       Scale applied to the original image
         * @param[in] spatial_scale  (Optional)Scale applied to the feature map. Defaults to 1.0
         * @param[in] pre_nms_topN   (Optional)Number of the best scores to be selected from the transformations. Defaults to 6000.
         * @param[in] post_nms_topN  (Optional)Number of the best scores to be selected from the NMS operation. Defaults to 300.
         * @param[in] nms_thres      (Optional)NMS overlap threshold. Defaults to 0.7.
         * @param[in] min_size       (Optional)Size used to validate the anchors produced. Defaults to 16.
         * @param[in] values_per_roi (Optional)Values used to represent a ROI(Region of interest). Defaults to 4.
         */
        BIGenerateProposalsInfo(float  im_width,
                                float  im_height,
                                float  im_scale,
                                float  spatial_scale  = 1.0,
                                int    pre_nms_topN   = 6000,
                                int    post_nms_topN  = 300,
                                float  nms_thres      = 0.7,
                                float  min_size       = 16.0,
                                size_t values_per_roi = 4)
            : _im_height(im_height),
              _im_width(im_width),
              _im_scale(im_scale),
              _spatial_scale(spatial_scale),
              _pre_nms_topN(pre_nms_topN),
              _post_nms_topN(post_nms_topN),
              _nms_thres(nms_thres),
              _min_size(min_size),
              _values_per_roi(values_per_roi)
        {
        }

        /* Get the original height */
        float im_height() const
        {
            return _im_height;
        }
        /* Get the original width */
        float im_width() const
        {
            return _im_width;
        }
        /* Get the image scale */
        float im_scale() const
        {
            return _im_scale;
        }
        /* Get the value of how many best scores to select (before NMS) */
        int pre_nms_topN() const
        {
            return _pre_nms_topN;
        }
        /* Get the value of how many best scores to select (after NMS) */
        int post_nms_topN() const
        {
            return _post_nms_topN;
        }
        /* Get the NMS overlap threshold */
        float nms_thres() const
        {
            return _nms_thres;
        }
        /* Get the minimal size */
        float min_size() const
        {
            return _min_size;
        }
        /* Get the spatial scale to be applied to the feature maps */
        float spatial_scale() const
        {
            return _spatial_scale;
        }
        /* Get the values used to represent a ROI(Region of interest)*/
        size_t values_per_roi() const
        {
            return _values_per_roi;
        }

    private:
        float  _im_height;
        float  _im_width;
        float  _im_scale;
        float  _spatial_scale;
        int    _pre_nms_topN;
        int    _post_nms_topN;
        float  _nms_thres;
        float  _min_size;
        size_t _values_per_roi;
    };

    /** Normalization Layer Information class */
    class BINormalizationLayerInfo
    {
    public:
        /** Default Constructor
         *
         * @param[in] type      The normalization type. Can be @ref NormType::IN_MAP_1D, @ref NormType::IN_MAP_2D or @ref NormType::CROSS_MAP
         * @param[in] norm_size The normalization size is the number of elements to normalize across. Defaults to 5.
         * @param[in] alpha     (Optional) Alpha parameter used by normalization equation. Defaults to 0.0001.
         * @param[in] beta      (Optional) Beta parameter used by normalization equation. Defaults to 0.5.
         * @param[in] kappa     (Optional) Kappa parameter used by [Krichevksy 2012] Across Channel Local Brightness Normalization equation.
         * @param[in] is_scaled (Optional) Boolean that specifies if alpha will be scaled by the normalization size or not.
         *                      Should be false to follow [Krichevksy 2012].
         */
        BINormalizationLayerInfo(BINormType type,
                                 uint32_t   norm_size = 5,
                                 float      alpha     = 0.0001f,
                                 float      beta      = 0.5f,
                                 float      kappa     = 1.f,
                                 bool       is_scaled = true)
            : _type(type), _norm_size(norm_size), _alpha(alpha), _beta(beta), _kappa(kappa), _is_scaled(is_scaled)
        {
        }
        /** Get the normalization type */
        BINormType type() const
        {
            return _type;
        }
        /** Get the normalization size */
        uint32_t norm_size() const
        {
            return _norm_size;
        }
        /** Get the alpha value */
        float alpha() const
        {
            return _alpha;
        }
        /** Get the beta value */
        float beta() const
        {
            return _beta;
        }
        /** Get the kappa value */
        float kappa() const
        {
            return _kappa;
        }
        /** Get the is_scaled value */
        bool is_scaled() const
        {
            return _is_scaled;
        }
        /** Check if normalization is cross map */
        bool is_cross_map() const
        {
            return _type == BINormType::CROSS_MAP;
        }
        /** Check if normalization is not cross map */
        bool is_in_map() const
        {
            return !is_cross_map();
        }
        /** Return the scaling factor of the normalization function.
         *
         * If is_scaled is set to false then [Krichevksy 2012] normalization scaling is performed,
         * where alpha is returned plainly, else alpha is scaled by the total number of elements used for the normalization.
         *
         * @return The normalization scaling factor.
         */
        float scale_coeff() const
        {
            const uint32_t size = (_type == BINormType::IN_MAP_2D) ? _norm_size * _norm_size : _norm_size;
            return (_is_scaled) ? (_alpha / size) : _alpha;
        }

    private:
        BINormType _type;
        uint32_t   _norm_size;
        float      _alpha;
        float      _beta;
        float      _kappa;
        bool       _is_scaled;
    };

    /** Pooling Layer Information struct*/
    struct BIPoolingLayerInfo
    {
        /** Default Constructor */
        BIPoolingLayerInfo()
            : pool_type(BIPoolingType::MAX),
              pool_size(Size2D()),
              data_layout(BIDataLayout::UNKNOWN),
              pad_stride_info(BIPadStrideInfo()),
              exclude_padding(false),
              is_global_pooling(false),
              fp_mixed_precision(false),
              use_inf_as_limit(true),
              use_kernel_indices(false)
        {
        }
        /** Constructor
         *
         * @param[in] pool_type          Pooling type @ref PoolingType.
         * @param[in] pool_size          Pooling size, in elements, across  x and y.
         * @param[in] data_layout        Data layout used by the layer @ref DataLayout
         * @param[in] pad_stride_info    (Optional) Padding and stride information @ref PadStrideInfo
         * @param[in] exclude_padding    (Optional) Strategy when accounting padding in calculations.
         *                               True will exclude padding while false will not (Used in AVG/L2 pooling to determine the pooling area).
         *                               Defaults to false;
         * @param[in] fp_mixed_precision (Optional) Use wider accumulators (32 bit instead of 16 for FP16) to improve accuracy.
         * @param[in] use_inf_as_limit   (Optional) Use inf to represent the limits of datatypes range, instead of  using "lowest" property of the data type.
         * @param[in] use_kernel_indices (Optional) Use kernel indices instead of using source indices while computing indices tensor.
         */
        explicit BIPoolingLayerInfo(BIPoolingType   pool_type,
                                    unsigned int    pool_size,
                                    BIDataLayout    data_layout,
                                    BIPadStrideInfo pad_stride_info    = BIPadStrideInfo(),
                                    bool            exclude_padding    = false,
                                    bool            fp_mixed_precision = false,
                                    bool            use_inf_as_limit   = true,
                                    bool            use_kernel_indices = false)
            : pool_type(pool_type),
              pool_size(Size2D(pool_size, pool_size)),
              data_layout(data_layout),
              pad_stride_info(pad_stride_info),
              exclude_padding(exclude_padding),
              is_global_pooling(false),
              fp_mixed_precision(fp_mixed_precision),
              use_inf_as_limit(use_inf_as_limit),
              use_kernel_indices(use_kernel_indices)
        {
        }

        /** Constructor
         *
         * @param[in] pool_type          Pooling type @ref PoolingType.
         * @param[in] pool_size          Pooling size, in elements, across  x and y.
         * @param[in] data_layout        Data layout used by the layer @ref DataLayout
         * @param[in] pad_stride_info    (Optional) Padding and stride information @ref PadStrideInfo
         * @param[in] exclude_padding    (Optional) Strategy when accounting padding in calculations.
         *                               True will exclude padding while false will not (Used in AVG/L2 pooling to determine the pooling area).
         *                               Defaults to false;
         * @param[in] fp_mixed_precision (Optional) Use wider accumulators (32 bit instead of 16 for FP16) to improve accuracy.
         * @param[in] use_inf_as_limit   (Optional) Use inf to represent the limits of datatypes range, instead of  using "lowest" property of the data type.
         * @param[in] use_kernel_indices (Optional) Use kernel indices instead of using source indices while computing indices tensor.
         */
        explicit BIPoolingLayerInfo(BIPoolingType   pool_type,
                                    Size2D          pool_size,
                                    BIDataLayout    data_layout,
                                    BIPadStrideInfo pad_stride_info    = BIPadStrideInfo(),
                                    bool            exclude_padding    = false,
                                    bool            fp_mixed_precision = false,
                                    bool            use_inf_as_limit   = true,
                                    bool            use_kernel_indices = false)
            : pool_type(pool_type),
              pool_size(pool_size),
              data_layout(data_layout),
              pad_stride_info(pad_stride_info),
              exclude_padding(exclude_padding),
              is_global_pooling(false),
              fp_mixed_precision(fp_mixed_precision),
              use_inf_as_limit(use_inf_as_limit),
              use_kernel_indices(use_kernel_indices)
        {
        }

        /** Constructor
         *
         * @note This constructor is used for global pooling
         *
         * @param[in] pool_type   Pooling type @ref PoolingType.
         * @param[in] data_layout Data layout used by the layer @ref DataLayout
         */
        explicit BIPoolingLayerInfo(BIPoolingType pool_type, BIDataLayout data_layout)
            : pool_type(pool_type),
              pool_size(Size2D()),
              data_layout(data_layout),
              pad_stride_info(BIPadStrideInfo(1, 1, 0, 0)),
              exclude_padding(false),
              is_global_pooling(true),
              fp_mixed_precision(false),
              use_inf_as_limit(true),
              use_kernel_indices(false)
        {
        }

        BIPoolingType   pool_type;
        Size2D          pool_size;
        BIDataLayout    data_layout;
        BIPadStrideInfo pad_stride_info;
        bool            exclude_padding;
        bool            is_global_pooling;
        bool            fp_mixed_precision;
        bool            use_inf_as_limit;
        bool            use_kernel_indices;
    };

    /** PriorBox layer info */
    class BIPriorBoxLayerInfo final
    {
    public:
        /** Default Constructor */
        BIPriorBoxLayerInfo()
            : _min_sizes(),
              _variances(),
              _offset(),
              _flip(true),
              _clip(false),
              _max_sizes(),
              _aspect_ratios(),
              _img_size(),
              _steps()
        {
        }
        /** Constructor
         *
         * @param[in] min_sizes     Min sizes vector.
         * @param[in] variances     Variances vector.
         * @param[in] offset        Offset value.
         * @param[in] flip          (Optional) Flip the aspect ratios.
         * @param[in] clip          (Optional) Clip coordinates so that they're within [0,1].
         * @param[in] max_sizes     (Optional) Max sizes vector.
         * @param[in] aspect_ratios (Optional) Aspect ratios of the boxes.
         * @param[in] img_size      (Optional) Image size.
         * @param[in] steps         (Optional) Step values.
         */
        BIPriorBoxLayerInfo(const std::vector<float>   &min_sizes,
                            const std::vector<float>   &variances,
                            float                       offset,
                            bool                        flip          = true,
                            bool                        clip          = false,
                            const std::vector<float>   &max_sizes     = {},
                            const std::vector<float>   &aspect_ratios = {},
                            const BICoordinates2D      &img_size      = BICoordinates2D{0, 0},
                            const std::array<float, 2> &steps         = {{0.f, 0.f}})
            : _min_sizes(min_sizes),
              _variances(variances),
              _offset(offset),
              _flip(flip),
              _clip(clip),
              _max_sizes(max_sizes),
              _aspect_ratios(),
              _img_size(img_size),
              _steps(steps)
        {
            _aspect_ratios.push_back(1.);
            for (unsigned int i = 0; i < aspect_ratios.size(); ++i)
            {
                float ar            = aspect_ratios[i];
                bool  already_exist = false;
                for (auto ar_new : _aspect_ratios)
                {
                    if (fabs(ar - ar_new) < 1e-6)
                    {
                        already_exist = true;
                        break;
                    }
                }
                if (!already_exist)
                {
                    _aspect_ratios.push_back(ar);
                    if (flip)
                    {
                        _aspect_ratios.push_back(1.f / ar);
                    }
                }
            }
        }
        /** Get min sizes. */
        std::vector<float> min_sizes() const
        {
            return _min_sizes;
        }
        /** Get min variances. */
        std::vector<float> variances() const
        {
            return _variances;
        }
        /** Get the step coordinates */
        std::array<float, 2> steps() const
        {
            return _steps;
        }
        /** Get the image size coordinates */
        BICoordinates2D img_size() const
        {
            return _img_size;
        }
        /** Get the offset */
        float offset() const
        {
            return _offset;
        }
        /** Get the flip value */
        bool flip() const
        {
            return _flip;
        }
        /** Get the clip value */
        bool clip() const
        {
            return _clip;
        }
        /** Get max sizes. */
        std::vector<float> max_sizes() const
        {
            return _max_sizes;
        }
        /** Get aspect ratios. */
        std::vector<float> aspect_ratios() const
        {
            return _aspect_ratios;
        }

    private:
        std::vector<float>   _min_sizes;
        std::vector<float>   _variances;
        float                _offset;
        bool                 _flip;
        bool                 _clip;
        std::vector<float>   _max_sizes;
        std::vector<float>   _aspect_ratios;
        BICoordinates2D      _img_size;
        std::array<float, 2> _steps;
    };

    /** ROI Pooling Layer Information class */
    class BIROIPoolingLayerInfo final
    {
    public:
        /** Constructor
         *
         * @param[in] pooled_width   Pooled width of the layer.
         * @param[in] pooled_height  Pooled height of the layer.
         * @param[in] spatial_scale  Spatial scale to be applied to the ROI coordinates and dimensions.
         * @param[in] sampling_ratio Number of samples to include in each pooling region (if set to zero, a ceil(roi_dims/pooling_dims))
         */
        BIROIPoolingLayerInfo(unsigned int pooled_width,
                              unsigned int pooled_height,
                              float        spatial_scale,
                              unsigned int sampling_ratio = 0)
            : _pooled_width(pooled_width),
              _pooled_height(pooled_height),
              _spatial_scale(spatial_scale),
              _sampling_ratio(sampling_ratio)
        {
        }
        /** Get the pooled width of the layer */
        unsigned int pooled_width() const
        {
            return _pooled_width;
        }
        /** Get the pooled height of the layer */
        unsigned int pooled_height() const
        {
            return _pooled_height;
        }
        /** Get the spatial scale */
        float spatial_scale() const
        {
            return _spatial_scale;
        }
        /** Get sampling ratio */
        unsigned int sampling_ratio() const
        {
            return _sampling_ratio;
        }

    private:
        unsigned int _pooled_width;
        unsigned int _pooled_height;
        float        _spatial_scale;
        unsigned int _sampling_ratio;
    };

    class BIStridedSliceLayerInfo
    {
    public:
        /** Default Constructor
         *
         * @param[in] begin_mask       (Optional) If the ith bit of begin_mask is set, starts[i] is ignored and the fullest possible range in that dimension is used instead.
         * @param[in] end_mask         (Optional) If the ith bit of end_mask is set, ends[i] is ignored and the fullest possible range in that dimension is used instead.
         * @param[in] shrink_axis_mask (Optional) If the ith bit of shrink_axis_mask is set, it implies that the ith specification shrinks the dimensionality by 1.
         */
        BIStridedSliceLayerInfo(int32_t begin_mask = 0, int32_t end_mask = 0, int32_t shrink_axis_mask = 0)
            : _begin_mask(begin_mask), _end_mask(end_mask), _shrink_axis_mask(shrink_axis_mask)
        {
        }

        /* Get the begin mask value */
        int32_t begin_mask() const
        {
            return _begin_mask;
        }

        /* Get the end mask value */
        int32_t end_mask() const
        {
            return _end_mask;
        }

        /* Get the shrink axis mask value */
        int32_t shrink_axis_mask() const
        {
            return _shrink_axis_mask;
        }

    private:
        int32_t _begin_mask;
        int32_t _end_mask;
        int32_t _shrink_axis_mask;
    };

}

#endif //BATMANINFER_BI_TYPES_HPP
