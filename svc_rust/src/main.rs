use actix_web::{get, http, web, App, HttpServer, Responder, Result};
use serde::Serialize;

use milvus::collection::{Collection, ParamValue, SearchOption};
use milvus::{client::Client, data::FieldColumn, error::Error, index::MetricType};

const INDEX_VEC_FIELD: &str = "embeddings";
const INDEX_HOST: &str = "http://localhost:19530";
const INDEX_NAME: &str = "tracks";
const INDEX_N_TOP: i32 = 10;
const INDEX_N_PROBE: i32 = 10;
const INDEX_ID_IDX: usize = 0;
const INDEX_VEC_IDX: usize = 1;
const INDEX_EXCLUDE_FIRST: bool = true;

const SERVICE_HOST: &str = "localhost";
const SERVICE_PORT: u16 = 8080;

#[derive(Serialize)]
struct TextResponse {
    message: String,
}

#[derive(Serialize)]
struct SimilarResponse {
    similar_ids: Vec<i64>,
}

async fn not_found() -> Result<impl Responder> {
    let resp = TextResponse {
        message: "Not found".to_string(),
    };
    Ok((web::Json(resp), http::StatusCode::NOT_FOUND))
}

async fn vec_by_id(
    id: i64,
    collection: &Collection,
) -> Result<Vec<FieldColumn>, milvus::error::Error> {
    collection
        .query::<_, [&str; 0]>(format!("id == {}", id), [])
        .await
}

async fn similar(target: i64) -> Result<Vec<i64>, Error> {
    let client = Client::new(INDEX_HOST).await?;
    let collection = client.get_collection(INDEX_NAME).await?;

    let query_response = vec_by_id(target, &collection).await?;
    let target_vec: Vec<f32> = query_response[INDEX_VEC_IDX]
        .value
        .clone()
        .try_into()
        .unwrap();

    let mut option = SearchOption::default();
    option.add_param("nprobe", ParamValue!(INDEX_N_PROBE));

    let result = collection
        .search(
            vec![target_vec.into()],
            INDEX_VEC_FIELD,
            INDEX_N_TOP + 1,
            MetricType::L2,
            vec!["id"],
            &option,
        )
        .await?;

    let finded_vec: Vec<i64> = result[0].field[INDEX_ID_IDX]
        .value
        .clone()
        .try_into()
        .unwrap();

    match INDEX_EXCLUDE_FIRST {
        true => Ok(finded_vec[1..].to_vec()),
        false => Ok(finded_vec[..INDEX_N_TOP as usize].to_vec()),
    }
}

#[get("/health")]
async fn healthcheck() -> Result<impl Responder> {
    let resp = TextResponse {
        message: "Service is up".to_string(),
    };
    Ok((web::Json(resp), http::StatusCode::OK))
}

#[get("/similar/{id}")]
async fn similar_handler(path: web::Path<i64>) -> Result<impl Responder> {
    let finded_vec = similar(path.into_inner()).await.unwrap();
    let resp = SimilarResponse {
        similar_ids: finded_vec,
    };
    Ok((web::Json(resp), http::StatusCode::OK))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .service(healthcheck)
            .service(similar_handler)
            .default_service(web::route().to(not_found))
    })
    .bind((SERVICE_HOST, SERVICE_PORT))?
    .run()
    .await
}
