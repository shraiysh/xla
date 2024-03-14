/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/spmd/collective_permute_iteration_instances.h"
#include <algorithm>
#include <limits>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/while_loop_analysis.h"

namespace xla {

absl::StatusOr<bool> CollectivePermuteIterationInstances::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  bool changed = false;

  // Find the collective permute instruction
  std::vector<HloInstruction *> collectivePermuteInstructions;
  for (HloComputation *C : module->computations(execution_threads)) {
    for (HloInstruction *I : C->instructions()) {
      if (I->opcode() == HloOpcode::kCollectivePermute) {
        auto pairs = I->source_target_pairs();

        int64_t max_device_num = -1;
        for(auto [source, target] : pairs) {
          max_device_num = std::max(std::max(source, target), max_device_num);
        }
        int64_t num_devices = max_device_num + 1;

        int64_t curridx = 0;
        while(curridx < pairs.size()) {
          auto pair = pairs[curridx];
          int64_t source = pair.first;
          int64_t target = pair.second;
          if(source == curridx && (source + 1 == target || (source == max_device_num && target == 0))) {
            ++curridx;
            continue;
          }
          VLOG(2) << "Cycle not found at " << source << ", " << target << ", i = " << curridx;
          return false;
        }

        if(curridx != num_devices) {
          VLOG(2) << "Not enough pairs: " << curridx << "!=" << num_devices;
          return false;
        }

        HloInstruction *whileOp = I->parent()->WhileCallInstruction();
        TF_ASSIGN_OR_RETURN(WhileLoopBackendConfig config,
                            whileOp->backend_config<WhileLoopBackendConfig>());
        int64_t n = config.known_trip_count().n();
        int64_t offset = n - num_devices;

        int64_t start = 1;
        std::vector<std::pair<int64_t, int64_t>> send_recv_validation;
        send_recv_validation.reserve(pairs.size());
        for (auto [a, b] : pairs) {
          send_recv_validation.push_back(
              std::make_pair(start, start + offset));
          start += 1;
        }

        VLOG(4) << "Source target instances : {";
        for (auto &[a, b] : send_recv_validation) {
          VLOG(4) << "{" << a << ", " << b << "}, ";
        }
        VLOG(4) << "}";

        xla::FrontendAttributes attributes;
        std::string iteration_instances =
            "{" +
            absl::StrJoin(send_recv_validation, ",",
                          absl::PairFormatter(
                              [](std::string *out, int64_t value) {
                                absl::StrAppend(out, "{", value);
                              },
                              ",",
                              [](std::string *out, int64_t value) {
                                absl::StrAppend(out, value, "}");
                              })) +
            "}";
        (*attributes.mutable_map())["_xla_send_recv_validation"] =
            iteration_instances;

        I->add_frontend_attributes(attributes);
        changed = true;
      }
    }
  }

  return changed;
}

}  // namespace xla
