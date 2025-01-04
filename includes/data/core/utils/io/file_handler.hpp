//
// Created by Mason on 2025/1/4.
//

#ifndef BATMANINFER_FILE_HANDLER_HPP
#define BATMANINFER_FILE_HANDLER_HPP

#include <fstream>
#include <string>

namespace BatmanInfer {
    namespace io {
        /** File Handling interface */
        class BIFileHandler {
        public:
            /** Default Constructor */
            BIFileHandler();

            /** Default Destructor */
            ~BIFileHandler();

            /** Allow instances of this class to be moved */
            BIFileHandler(BIFileHandler &&) = default;

            /** Prevent instances of this class from being copied (As this class contains pointers) */
            BIFileHandler(const BIFileHandler &) = delete;

            /** Prevent instances of this class from being copied (As this class contains pointers) */
            BIFileHandler &operator=(const BIFileHandler &) = delete;

            /** Allow instances of this class to be moved */
            BIFileHandler &operator=(BIFileHandler &&) = default;

            /** Opens file
             *
             * @param[in] filename File name
             * @param[in] mode     File open mode
             */
            void open(const std::string &filename, std::ios_base::openmode mode);

            /** Closes file */
            void close();

            /** Returns the file stream
             *
             * @return File stream
             */
            std::fstream &stream();

            /** Returns filename of the handled file
             *
             * @return File filename
             */
            std::string filename() const;

        private:
            std::fstream            _filestream;
            std::string             _filename;
            std::ios_base::openmode _mode;
        };
    }
}

#endif //BATMANINFER_FILE_HANDLER_HPP
