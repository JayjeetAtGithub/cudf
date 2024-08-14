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

#include "utils.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/io/parquet.hpp>

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>

#include <cudf_benchmark/tpch_datagen.hpp>

/**
 * @brief Write a cudf::table to a parquet file
 *
 * @param table The cudf::table to write
 * @param path The path to write the parquet file to
 * @param col_names The names of the columns in the table
 */
void write_parquet(std::unique_ptr<cudf::table> table,
                   std::string const& path,
                   std::vector<std::string> const& col_names)
{
  CUDF_FUNC_RANGE();
  std::cout << __func__ << " : " << path << std::endl;
  cudf::io::table_metadata metadata;
  std::vector<cudf::io::column_name_info> col_name_infos;
  for (auto& col_name : col_names) {
    col_name_infos.push_back(cudf::io::column_name_info(col_name));
  }
  metadata.schema_info            = col_name_infos;
  auto const table_input_metadata = cudf::io::table_input_metadata{metadata};
  auto builder = cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info(path));
  builder.metadata(table_input_metadata);
  auto const options = builder.build();
  cudf::io::parquet_chunked_writer(options).write(table->view());
}

int main(int argc, char** argv)
{
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " [scale_factor]"
              << " [memory_resource_type]" << std::endl;
    return 1;
  }

  cudf::size_type scale_factor = std::atoi(argv[1]);
  std::cout << "Generating scale factor: " << scale_factor << std::endl;

  std::string memory_resource_type = argv[2];
  auto resource                    = create_memory_resource(memory_resource_type);
  rmm::mr::set_current_device_resource(resource.get());

  auto const [free, total] = rmm::available_device_memory();
  std::cout << "Total GPU memory: " << total << std::endl;
  std::cout << "Available GPU memory: " << free << std::endl;

  auto const mem_stats_logger = memory_stats_logger();

  auto [orders, lineitem, part] = cudf::datagen::generate_orders_lineitem_part(
    scale_factor, cudf::get_default_stream(), rmm::mr::get_current_device_resource());
  write_parquet(std::move(orders), "orders.parquet", cudf::datagen::schema::ORDERS);
  write_parquet(std::move(lineitem), "lineitem.parquet", cudf::datagen::schema::LINEITEM);
  write_parquet(std::move(part), "part.parquet", cudf::datagen::schema::PART);

  auto partsupp = cudf::datagen::generate_partsupp(
    scale_factor, cudf::get_default_stream(), rmm::mr::get_current_device_resource());
  write_parquet(std::move(partsupp), "partsupp.parquet", cudf::datagen::schema::PARTSUPP);

  auto supplier = cudf::datagen::generate_supplier(
    scale_factor, cudf::get_default_stream(), rmm::mr::get_current_device_resource());
  write_parquet(std::move(supplier), "supplier.parquet", cudf::datagen::schema::SUPPLIER);

  auto customer = cudf::datagen::generate_customer(
    scale_factor, cudf::get_default_stream(), rmm::mr::get_current_device_resource());
  write_parquet(std::move(customer), "customer.parquet", cudf::datagen::schema::CUSTOMER);

  auto nation = cudf::datagen::generate_nation(cudf::get_default_stream(),
                                               rmm::mr::get_current_device_resource());
  write_parquet(std::move(nation), "nation.parquet", cudf::datagen::schema::NATION);

  auto region = cudf::datagen::generate_region(cudf::get_default_stream(),
                                               rmm::mr::get_current_device_resource());
  write_parquet(std::move(region), "region.parquet", cudf::datagen::schema::REGION);
  mem_stats_logger.print_peak_memory_usage();
}
