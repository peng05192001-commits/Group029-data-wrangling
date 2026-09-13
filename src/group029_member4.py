from __future__ import annotations

from pathlib import Path
import json
import re
import xml.etree.ElementTree as ET

import pandas as pd

DELIVERY_COLUMNS = [
    'delivery_id','order_id','dispatch_date','promised_date','delivered_date',
    'carrier','service_level','delivery_status','delay_days','on_time_in_full',
    'fulfilment_hours','delivery_cost','delay_reason','promised_days',
    'tracking_event_count','delivery_window','shipping_distance_km',
    'signature_required','estimated_carbon_kg','delivery_note_clean'
]


def _clean_text(value) -> str:
    if value is None:
        raise ValueError('Required delivery text value is missing')
    text = str(value).strip()
    if text == '' or text.lower() in {'nan','null'}:
        raise ValueError('Required delivery text value is empty/null-like')
    return text


def _parse_json_date(value) -> str:
    dt = pd.to_datetime(_clean_text(value), format='%Y-%m-%d', errors='raise')
    return dt.strftime('%Y-%m-%d')


def _parse_xml_date(value) -> str:
    dt = pd.to_datetime(_clean_text(value), format='%d/%m/%Y', errors='raise')
    return dt.strftime('%Y-%m-%d')


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = _clean_text(value).upper()
    if text in {'Y','TRUE'}:
        return True
    if text in {'N','FALSE'}:
        return False
    raise ValueError(f'Unexpected boolean value: {value!r}')


def _parse_int(value) -> int:
    if isinstance(value, bool):
        raise ValueError('Boolean is not a valid integer value')
    number = pd.to_numeric(value, errors='raise')
    if pd.isna(number) or float(number) % 1 != 0:
        raise ValueError(f'Expected integer-like value: {value!r}')
    return int(number)


def _parse_float(value) -> float:
    if isinstance(value, str):
        value = re.sub(r'^AUD\s*', '', value.strip(), flags=re.IGNORECASE).replace(',', '')
    number = pd.to_numeric(value, errors='raise')
    if pd.isna(number):
        raise ValueError('Required numeric delivery value is missing')
    return float(number)


def _json_delivery_row(order: dict) -> dict:
    d = order['delivery']
    return {
        'delivery_id': _clean_text(d['deliveryID']),
        'order_id': _clean_text(d['orderID']),
        'dispatch_date': _parse_json_date(d['dispatchDate']),
        'promised_date': _parse_json_date(d['promisedDate']),
        'delivered_date': _parse_json_date(d['deliveredDate']),
        'carrier': _clean_text(d['carrier']),
        'service_level': _clean_text(d['serviceLevel']),
        'delivery_status': _clean_text(d['deliveryStatus']),
        'delay_days': _parse_int(d['delayDays']),
        'on_time_in_full': _parse_bool(d['onTimeInFull']),
        'fulfilment_hours': _parse_int(d['fulfilmentHours']),
        'delivery_cost': _parse_float(d['deliveryCost']),
        'delay_reason': _clean_text(d['delayReason']),
        'promised_days': _parse_int(d['promisedDays']),
        'tracking_event_count': _parse_int(d['trackingEventCount']),
        'delivery_window': _clean_text(d['deliveryWindow']),
        'shipping_distance_km': _parse_float(d['shippingDistanceKm']),
        'signature_required': _parse_bool(d['signatureRequired']),
        'estimated_carbon_kg': _parse_float(d['estimatedCarbonKg']),
        'delivery_note_clean': _clean_text(d['deliveryNoteClean']),
    }


def _xml_delivery_row(order_elem: ET.Element) -> dict:
    d = order_elem.find('Delivery')
    if d is None:
        raise ValueError('XML Order is missing Delivery element')
    def t(tag: str):
        elem = d.find(tag)
        if elem is None:
            raise ValueError(f'XML Delivery missing {tag}')
        return elem.text
    return {
        'delivery_id': _clean_text(t('Delivery_ID')),
        'order_id': _clean_text(t('Order_ID')),
        'dispatch_date': _parse_xml_date(t('Dispatch_Date')),
        'promised_date': _parse_xml_date(t('Promised_Date')),
        'delivered_date': _parse_xml_date(t('Delivered_Date')),
        'carrier': _clean_text(t('Carrier')),
        'service_level': _clean_text(t('Service_Level')),
        'delivery_status': _clean_text(t('Delivery_Status')),
        'delay_days': _parse_int(t('Delay_Days')),
        'on_time_in_full': _parse_bool(t('On_Time_In_Full')),
        'fulfilment_hours': _parse_int(t('Fulfilment_Hours')),
        'delivery_cost': _parse_float(t('Delivery_Cost')),
        'delay_reason': _clean_text(t('Delay_Reason')),
        'promised_days': _parse_int(t('Promised_Days')),
        'tracking_event_count': _parse_int(t('Tracking_Event_Count')),
        'delivery_window': _clean_text(t('Delivery_Window')),
        'shipping_distance_km': _parse_float(t('Shipping_Distance_Km')),
        'signature_required': _parse_bool(t('Signature_Required')),
        'estimated_carbon_kg': _parse_float(t('Estimated_Carbon_Kg')),
        'delivery_note_clean': _clean_text(t('Delivery_Note_Clean')),
    }


def _conflict_keys(df: pd.DataFrame, key: str = 'delivery_id') -> list[str]:
    value_cols = [c for c in DELIVERY_COLUMNS if c != key]
    conflicts = []
    for k, group in df.groupby(key, sort=False):
        if len(group[value_cols].drop_duplicates()) > 1:
            conflicts.append(k)
    return conflicts


def build_deliveries(json_data: dict, xml_root: ET.Element):
    json_rows = [_json_delivery_row(order) for order in json_data['orders'] if order.get('delivery') is not None]
    xml_rows = [_xml_delivery_row(order) for order in xml_root.findall('./Orders/Order') if order.find('Delivery') is not None]

    json_df = pd.DataFrame(json_rows, columns=DELIVERY_COLUMNS)
    xml_df = pd.DataFrame(xml_rows, columns=DELIVERY_COLUMNS)

    json_conflicts = _conflict_keys(json_df)
    xml_conflicts = _conflict_keys(xml_df)
    if json_conflicts or xml_conflicts:
        raise ValueError(f'Within-source delivery conflicts: JSON={json_conflicts[:5]}, XML={xml_conflicts[:5]}')

    json_unique = json_df.drop_duplicates('delivery_id', keep='first').copy()
    xml_unique = xml_df.drop_duplicates('delivery_id', keep='first').copy()

    combined = pd.concat([
        json_unique.assign(_source='JSON'),
        xml_unique.assign(_source='XML')
    ], ignore_index=True)
    cross_conflicts = _conflict_keys(combined.drop(columns='_source'))
    if cross_conflicts:
        raise ValueError(f'Cross-source delivery conflicts: {cross_conflicts[:10]}')

    deliveries = (
        combined.drop(columns='_source')
        .drop_duplicates('delivery_id', keep='first')
        .sort_values('delivery_id')
        .reset_index(drop=True)
    )

    # Explicit dtypes for the published contract.
    for col in ['delivery_id','order_id','dispatch_date','promised_date','delivered_date','carrier',
                'service_level','delivery_status','delay_reason','delivery_window','delivery_note_clean']:
        deliveries[col] = deliveries[col].astype('string')
    for col in ['delay_days','fulfilment_hours','promised_days','tracking_event_count']:
        deliveries[col] = deliveries[col].astype('int64')
    for col in ['delivery_cost','shipping_distance_km','estimated_carbon_kg']:
        deliveries[col] = deliveries[col].astype('float64')
    for col in ['on_time_in_full','signature_required']:
        deliveries[col] = deliveries[col].astype(bool)

    profile = pd.DataFrame([
        {'source':'JSON','input_rows':len(json_df),'unique_delivery_ids':json_df['delivery_id'].nunique(),
         'within_source_duplicate_rows':len(json_df)-json_df['delivery_id'].nunique()},
        {'source':'XML','input_rows':len(xml_df),'unique_delivery_ids':xml_df['delivery_id'].nunique(),
         'within_source_duplicate_rows':len(xml_df)-xml_df['delivery_id'].nunique()},
    ])
    profile['cross_source_overlap_ids'] = len(set(json_unique['delivery_id']) & set(xml_unique['delivery_id']))
    profile['canonical_rows'] = len(deliveries)

    return {
        'deliveries': deliveries,
        'json_normalised': json_df,
        'xml_normalised': xml_df,
        'source_profile': profile,
        'within_json_conflicts': json_conflicts,
        'within_xml_conflicts': xml_conflicts,
        'cross_source_conflicts': cross_conflicts,
    }


def validate_deliveries(member4_tables: dict, orders: pd.DataFrame) -> pd.DataFrame:
    d = member4_tables['deliveries']
    records = []
    def add(vid, area, check, observed, passed, interpretation):
        records.append({'validation_id':vid,'area':area,'check':check,'observed_result':str(observed),
                        'status':'PASS' if passed else 'FAIL','passed':bool(passed),'interpretation':interpretation})

    add('VAL-SCHEMA-DEL-01','schema','deliveries columns and order', d.columns.tolist(),
        d.columns.tolist()==DELIVERY_COLUMNS, 'Matches the public data dictionary field order.')
    add('VAL-MISS-DEL-01','missingness','no pandas missing values in required delivery fields', int(d.isna().sum().sum()),
        int(d.isna().sum().sum())==0, 'All 20 delivery fields are non-nullable in the public contract.')
    add('VAL-PK-DEL-01','keys','delivery_id complete and unique',
        f"missing={int(d['delivery_id'].isna().sum())}; duplicates={int(d['delivery_id'].duplicated().sum())}",
        d['delivery_id'].notna().all() and d['delivery_id'].is_unique, 'One canonical row is retained per delivery_id.')
    missing_orders = sorted(set(d['order_id']) - set(orders['order_id']))
    orders_without_delivery = sorted(set(orders['order_id']) - set(d['order_id']))
    add('VAL-FK-DEL-01','relationships','delivery order_id resolves to orders.order_id and every completed order has one delivery',
        f'missing_order_refs={len(missing_orders)}; completed_orders_without_delivery={len(orders_without_delivery)}',
        len(missing_orders)==0 and len(orders_without_delivery)==0 and d['order_id'].is_unique,
        'Deliveries and completed orders form the required one-to-one relationship for this package.')

    expected_types = (
        pd.api.types.is_bool_dtype(d['on_time_in_full']) and pd.api.types.is_bool_dtype(d['signature_required']) and
        all(pd.api.types.is_integer_dtype(d[c]) for c in ['delay_days','fulfilment_hours','promised_days','tracking_event_count']) and
        all(pd.api.types.is_float_dtype(d[c]) for c in ['delivery_cost','shipping_distance_km','estimated_carbon_kg'])
    )
    add('VAL-TYPE-DEL-01','schema','delivery boolean/integer/numeric dtypes', d.dtypes.astype(str).to_dict(), expected_types,
        'Published booleans and numeric fields use typed values rather than raw source strings.')

    profile = member4_tables['source_profile']
    add('VAL-FLOW-DEL-01','flow','source row flow, within-source duplicates, overlap and canonical rows', profile.to_dict('records'),
        profile['canonical_rows'].nunique()==1 and int(profile['canonical_rows'].iloc[0])==d['delivery_id'].nunique(),
        'Counts are derived from the allocated JSON/XML; duplicates and overlap are reconciled by delivery_id.')
    conflict_count = (len(member4_tables['within_json_conflicts']) + len(member4_tables['within_xml_conflicts']) + len(member4_tables['cross_source_conflicts']))
    add('VAL-FLOW-DEL-02','flow','normalised duplicate/overlap field conflicts', conflict_count, conflict_count==0,
        'No differing normalised values are silently overwritten.')

    dispatch = pd.to_datetime(d['dispatch_date'], format='%Y-%m-%d')
    promised = pd.to_datetime(d['promised_date'], format='%Y-%m-%d')
    delivered = pd.to_datetime(d['delivered_date'], format='%Y-%m-%d')
    order_dates = pd.to_datetime(orders.set_index('order_id').loc[d['order_id'], 'order_timestamp'].to_numpy()).normalize()
    temporal_bad = int((order_dates > dispatch).sum() + (dispatch > promised).sum() + (dispatch > delivered).sum())
    add('VAL-TIME-DEL-01','temporal','order date <= dispatch; dispatch <= promised and delivered', temporal_bad, temporal_bad==0,
        'Checks the required temporal ordering without treating late delivery after the promise date as invalid.')

    expected_delay = (delivered-promised).dt.days.clip(lower=0).astype(int)
    expected_promised = (promised-dispatch).dt.days.astype(int)
    arithmetic_bad = int((d['delay_days']!=expected_delay).sum() + (d['promised_days']!=expected_promised).sum())
    add('VAL-ARITH-DEL-01','arithmetic','delay_days and promised_days agree with dates', arithmetic_bad, arithmetic_bad==0,
        'delay_days=max(delivered-promised,0); promised_days=promised-dispatch.')
    otif_bad = int((d['on_time_in_full'] != d['delay_days'].eq(0)).sum())
    add('VAL-ARITH-DEL-02','arithmetic','on_time_in_full agrees with delay_days', otif_bad, otif_bad==0,
        'The package encodes OTIF consistently with zero delivery delay.')

    valid_ranges = (
        d['delay_days'].ge(0).all() and d['fulfilment_hours'].ge(0).all() and d['delivery_cost'].ge(0).all() and
        d['promised_days'].ge(0).all() and d['tracking_event_count'].ge(0).all() and
        d['shipping_distance_km'].ge(0).all() and d['estimated_carbon_kg'].ge(0).all()
    )
    add('VAL-RANGE-DEL-01','ranges','non-negative delivery numeric ranges', bool(valid_ranges), bool(valid_ranges),
        'Operational count, cost, distance, carbon and delay measures are non-negative.')

    result = pd.DataFrame(records)
    return result


def write_delivery_output(deliveries: pd.DataFrame, output_dir: Path, group_id: str='Group029') -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f'{group_id}_deliveries_standardised.csv'
    deliveries.to_csv(path, index=False, encoding='utf-8')
    return path
