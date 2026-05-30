create table train_schedules (
  id bigserial primary key,
  train_no text,
  train_name text,
  route text,
  station_order int,
  station_name text,
  station_code text,
  arrival_time text,
  departure_time text,
  note text,
  source_page int
);