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

#include "../utilities/timer.hpp"
#include "utils.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>

#include <cudf_benchmark/tpch_datagen.hpp>

/**
 * @file q10.cpp
 * @brief Implement query 10 of the TPC-H benchmark.
 *
 * create view customer as select * from '/tables/scale-1/customer.parquet';
 * create view orders as select * from '/tables/scale-1/orders.parquet';
 * create view lineitem as select * from '/tables/scale-1/lineitem.parquet';
 * create view nation as select * from '/tables/scale-1/nation.parquet';
 *
 * select
 *    c_custkey,
 *    c_name,
 *    sum(l_extendedprice * (1 - l_discount)) as revenue,
 *    c_acctbal,
 *    n_name,
 *    c_address,
 *    c_phone,
 *    c_comment
 * from
 *    customer,
 *    orders,
 *    lineitem,
 *    nation
 * where
 *     c_custkey = o_custkey
 *     and l_orderkey = o_orderkey
 *     and o_orderdate >= date '1993-10-01'
 *     and o_orderdate < date '1994-01-01'
 *     and l_returnflag = 'R'
 *     and c_nationkey = n_nationkey
 * group by
 *     c_custkey,
 *     c_name,
 *     c_acctbal,
 *     c_phone,
 *     n_name,
 *     c_address,
 *     c_comment
 * order by
 *     revenue desc;
 */

/**
 * @brief Calculate the revenue column
 *
 * @param extendedprice The extended price column
 * @param discount The discount column
 * @param stream The CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 */
[[nodiscard]] std::unique_ptr<cudf::column> calc_revenue(
  cudf::column_view const& extendedprice,
  cudf::column_view const& discount,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource())
{
  auto const one = cudf::numeric_scalar<double>(1);
  auto const one_minus_discount =
    cudf::binary_operation(one, discount, cudf::binary_operator::SUB, discount.type(), stream, mr);
  auto const revenue_type = cudf::data_type{cudf::type_id::FLOAT64};
  auto revenue            = cudf::binary_operation(extendedprice,
                                        one_minus_discount->view(),
                                        cudf::binary_operator::MUL,
                                        revenue_type,
                                        stream,
                                        mr);
  return revenue;
}

/**
 * @brief Generate or read the dataset
 *
 * @param source The dataset source
 */
[[nodiscard]] auto prepare_dataset(std::string source)
{
  // Define the column projection and filter predicates for the source tables
  std::vector<std::string> const orders_cols   = {"o_custkey", "o_orderkey", "o_orderdate"};
  std::vector<std::string> const lineitem_cols = {
    "l_extendedprice", "l_discount", "l_orderkey", "l_returnflag"};
  std::vector<std::string> const customer_cols = {
    "c_custkey", "c_name", "c_nationkey", "c_acctbal", "c_address", "c_phone", "c_comment"};
  std::vector<std::string> const nation_cols = {"n_name", "n_nationkey"};

  auto const o_orderdate_ref = cudf::ast::column_reference(std::distance(
    orders_cols.begin(), std::find(orders_cols.begin(), orders_cols.end(), "o_orderdate")));
  auto o_orderdate_lower =
    cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1993, 10, 1), true);
  auto const o_orderdate_lower_limit = cudf::ast::literal(o_orderdate_lower);
  auto const o_orderdate_pred_lower  = cudf::ast::operation(
    cudf::ast::ast_operator::GREATER_EQUAL, o_orderdate_ref, o_orderdate_lower_limit);
  auto o_orderdate_upper =
    cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1994, 1, 1), true);
  auto const o_orderdate_upper_limit = cudf::ast::literal(o_orderdate_upper);
  auto const o_orderdate_pred_upper =
    cudf::ast::operation(cudf::ast::ast_operator::LESS, o_orderdate_ref, o_orderdate_upper_limit);
  auto const orders_pred = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::LOGICAL_AND, o_orderdate_pred_lower, o_orderdate_pred_upper);

  auto const l_returnflag_ref = cudf::ast::column_reference(3);
  auto r_scalar               = cudf::string_scalar("R");
  auto const r_literal        = cudf::ast::literal(r_scalar);
  auto const lineitem_pred    = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::EQUAL, l_returnflag_ref, r_literal);

  if (source == "cudf_datagen") {
    auto [o, l, p] = cudf::datagen::generate_orders_lineitem_part(
      get_sf(), cudf::get_default_stream(), rmm::mr::get_current_device_resource());
    auto orders = std::make_unique<table_with_names>(std::move(o), cudf::datagen::schema::ORDERS);
    auto orders_projected = apply_projection(std::move(orders), orders_cols);
    auto orders_filtered  = apply_filter(std::move(orders_projected), *orders_pred);

    auto lineitem =
      std::make_unique<table_with_names>(std::move(l), cudf::datagen::schema::LINEITEM);
    auto lineitem_projected = apply_projection(std::move(lineitem), lineitem_cols);
    auto lineitem_filtered  = apply_filter(std::move(lineitem_projected), *lineitem_pred);

    auto c = cudf::datagen::generate_customer(
      get_sf(), cudf::get_default_stream(), rmm::mr::get_current_device_resource());
    auto customer =
      std::make_unique<table_with_names>(std::move(c), cudf::datagen::schema::CUSTOMER);
    auto customer_projected = apply_projection(std::move(customer), customer_cols);

    auto n      = cudf::datagen::generate_nation(cudf::get_default_stream(),
                                            rmm::mr::get_current_device_resource());
    auto nation = std::make_unique<table_with_names>(std::move(n), cudf::datagen::schema::NATION);
    auto nation_projected = apply_projection(std::move(nation), nation_cols);

    return std::make_tuple(std::move(orders_filtered),
                           std::move(lineitem_filtered),
                           std::move(customer_projected),
                           std::move(nation_projected));
  } else {
    auto orders = read_parquet(source + "/orders.parquet", orders_cols, std::move(orders_pred));
    auto lineitem =
      read_parquet(source + "/lineitem.parquet", lineitem_cols, std::move(lineitem_pred));
    auto customer = read_parquet(source + "/customer.parquet", customer_cols);
    auto nation   = read_parquet(source + "/nation.parquet", nation_cols);

    return std::make_tuple(
      std::move(orders), std::move(lineitem), std::move(customer), std::move(nation));
  }
}

int main(int argc, char const** argv)
{
  auto const args = parse_args(argc, argv);

  // Create memory resource
  auto resource = create_memory_resource(args.memory_resource_type);
  rmm::mr::set_current_device_resource(resource.get());

  // Print hardware stats
  print_hardware_stats();

  // Instantiate the memory stats logger
  auto const mem_stats_logger = memory_stats_logger();

  // Start timer
  cudf::examples::timer timer;

  // Prepare the dataset
  auto [orders, lineitem, customer, nation] = prepare_dataset(args.dataset_dir);

  // Perform the joins
  auto const join_a       = apply_inner_join(customer, nation, {"c_nationkey"}, {"n_nationkey"});
  auto const join_b       = apply_inner_join(lineitem, orders, {"l_orderkey"}, {"o_orderkey"});
  auto const joined_table = apply_inner_join(join_a, join_b, {"c_custkey"}, {"o_custkey"});

  // Calculate and append the `revenue` column
  auto revenue =
    calc_revenue(joined_table->column("l_extendedprice"), joined_table->column("l_discount"));
  (*joined_table).append(revenue, "revenue");

  // Perform the groupby operation
  auto const groupedby_table = apply_groupby(
    joined_table,
    groupby_context_t{
      {"c_custkey", "c_name", "c_acctbal", "c_phone", "n_name", "c_address", "c_comment"},
      {
        {"revenue", {{cudf::aggregation::Kind::SUM, "revenue"}}},
      }});

  // Perform the order by operation
  auto const orderedby_table =
    apply_orderby(groupedby_table, {"revenue"}, {cudf::order::DESCENDING});

  // End timer and print elapsed time
  timer.print_elapsed_millis();

  // Print the peak memory usage
  mem_stats_logger.print_peak_memory_usage();

  // Write query result to a parquet file
  orderedby_table->to_parquet("q10.parquet");
  return 0;
}
