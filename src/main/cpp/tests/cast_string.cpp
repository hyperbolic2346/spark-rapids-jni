/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cast_string.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <rmm/device_uvector.hpp>

struct StringToIntegerTests : public cudf::test::BaseFixture {
};

TEST_F(StringToIntegerTests, Simple)
{
  using namespace cudf;

  auto const strings = test::strings_column_wrapper{"1", "0", "42"};
  strings_column_view scv{strings};

  auto const result = spark_rapids_jni::string_to_int32(scv, false, rmm::cuda_stream_default);

  cudf::test::fixed_width_column_wrapper<int32_t> expected({1, 0, 42}, {1, 1, 1});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}