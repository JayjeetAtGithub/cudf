/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf/interop.hpp>

arrow::BinaryViewType::c_type create_inline_binary_view(std::string_view data)
{
  arrow::BinaryViewType::c_type view;
  view.inlined = {static_cast<int32_t>(data.size()), {}};
  // Inlined is a fixed size array of 12 bytes
  memcpy(&view.inlined.data, data.data(), data.size());
  return view;
}

arrow::BinaryViewType::c_type create_binary_view(std::string_view data,
                                                 int32_t buffer_idx,
                                                 int32_t buffer_offset)
{
  arrow::BinaryViewType::c_type view;
  view.ref = {static_cast<int32_t>(data.size()), {}, buffer_idx, buffer_offset};
  // Prefix is a fixed size array of 4 bytes
  memcpy(&view.ref.prefix, data.data(), sizeof(view.ref.prefix));
  return view;
}

std::shared_ptr<arrow::StringViewArray> create_string_view_col(
  std::vector<std::shared_ptr<arrow::Buffer>> data_buffers,
  std::vector<arrow::BinaryViewType::c_type> views)
{
  // The length of the column
  auto const length = static_cast<int64_t>(views.size());
  auto array        = std::make_shared<arrow::StringViewArray>(
    arrow::utf8_view(), length, arrow::Buffer::FromVector(views), std::move(data_buffers));
  array->ValidateFull();
  return array;
}

std::unique_ptr<cudf::scalar> convert_string_arrow_to_cudf()
{
  auto const value        = std::string("Arrow String");
  auto const arrow_scalar = arrow::StringScalar(value);
  auto cudf_scalar        = cudf::from_arrow(arrow_scalar);
  return cudf_scalar;
}

int main(int argc, char** argv)
{
  convert_string_arrow_to_cudf();

  // Create a string view col
  std::vector<std::shared_ptr<arrow::Buffer>> data_buffers;
  std::vector<arrow::BinaryViewType::c_type> views;

  auto buffer_a = arrow::Buffer::FromString("helloworldapachearrowcudfnvidia");
  data_buffers.push_back(buffer_a);

  views.push_back(create_binary_view("helloworld", 0, 0));
  views.push_back(create_binary_view("apachearrow", 0, 10));
  views.push_back(create_inline_binary_view("cudf"));

  auto string_view_col = create_string_view_col(data_buffers, views);
  for (int i = 0; i < string_view_col->length(); i++) {
    std::cout << string_view_col->GetString(i) << std::endl;
  }
}
