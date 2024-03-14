/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/while_loop_trip_count_annotator.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace {

using CollectivePermuteIterationInstancesTest = HloTestBase;
namespace op = xla::testing::opcode_matchers;

TEST_F(CollectivePermuteIterationInstancesTest, SimpleMove) {
  absl::string_view hlo_string = R"(
HloModule test
Body {
  param = (s32[]) parameter(0)
  i = s32[] get-tuple-element(param), index=0
  one = s32[] constant(1)
  i_plus_one = s32[] add(i, one)
  permute = s32[] collective-permute(i_plus_one), source_target_pairs={{0,1}, {1,2}, {2,3}, {3,0}}
  ROOT tuple = (s32[]) tuple(i_plus_one)
}

%Cond (param.1: (s32[])) -> pred[] {
  %param.1 = (s32[]) parameter(0)
  %i.1 = s32[] get-tuple-element((s32[]) %param.1), index=0
  %trip_count = s32[] constant(11)
  ROOT %done = pred[] compare(s32[] %i.1, s32[] %trip_count), direction=LT
}

ENTRY test {
  i_start = s32[] constant(0)
  initial_tuple = (s32[]) tuple(i_start)
  ROOT while = (s32[]) while(initial_tuple), condition=Cond, body=Body
}
  )";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  HloPassPipeline myPipeline("my-pass-pipeline");
  myPipeline.AddPass<WhileLoopTripCountAnnotator>();
  myPipeline.AddPass<CollectivePermuteIterationInstances>();
  ASSERT_TRUE(myPipeline.Run(&*module).value());

  std::cerr << module->ToString() << "\n";

}

}  // namespace
}  // namespace xla
