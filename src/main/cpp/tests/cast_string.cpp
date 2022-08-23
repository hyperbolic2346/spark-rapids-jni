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

using namespace cudf;

struct StringToIntegerTests : public test::BaseFixture {
};

TEST_F(StringToIntegerTests, Simple)
{
  auto const strings = test::strings_column_wrapper{"1", "0", "42"};
  strings_column_view scv{strings};

  auto const result = spark_rapids_jni::string_to_int32(scv, false, rmm::cuda_stream_default);

  test::fixed_width_column_wrapper<int32_t> expected({1, 0, 42}, {1, 1, 1});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringToIntegerTests, Ansi)
{
  auto const strings = test::strings_column_wrapper({"+1", "-0", "4.2", "asdf", "98fe", "  00012", ".--e-37602.n",
                                                    "\r\r\t\n11.12380", "-.2", ".3", ".", "+1.2", "\n123\n456\n",
                                                    "1 2", "null", "123", "", "1. 2", "+    7.6"}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1});
  strings_column_view scv{strings};

  try {
    spark_rapids_jni::string_to_int32(scv, true, rmm::cuda_stream_default);
  } catch(logic_error &e) {
    auto const error_msg = e.what();
    auto const len = strlen(error_msg);
    // compare last n chars
    const char *expected_end = "String column had 9 parse errors, first error was line 4: 'asdf'!";
    auto const expected_end_len = strlen(expected_end);

    EXPECT_STREQ(error_msg + (len - expected_end_len), expected_end);
  }

  auto const result = spark_rapids_jni::string_to_int32(scv, false, rmm::cuda_stream_default);

  test::fixed_width_column_wrapper<int32_t> expected({1, 0, 4, 0, 0, 12, 0, 11, 0, 0, 0, 1, 0, 0, 0, 123, 0, 0, 0}, {1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}