#pragma once

#include "core/config/config.h"
#include "ofm.h"

namespace ofm {
void InitOFMAsync(OFM& _ofm, const OFMConfiguration& _config, cudaStream_t _stream);
}