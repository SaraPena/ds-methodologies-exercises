use zillow;

Select p.*, logerror, MAX(transactiondate), airconditioningdesc, architecturalstyledesc, buildingclassdesc, heatingorsystemdesc, propertylandusedesc,storydesc, typeconstructiondesc
FROM predictions_2017
LEFT JOIN properties_2017 as p using (parcelid)
LEFT JOIN airconditioningtype using (airconditioningtypeid)
LEFT JOIN architecturalstyletype using (architecturalstyletypeid)
LEFT JOIN buildingclasstype using (buildingclasstypeid)
LEFT JOIN propertylandusetype using (propertylandusetypeid)
LEFT JOIN storytype using (storytypeid)
LEFT JOIN typeconstructiontype using (typeconstructiontypeid)
LEFT JOIN heatingorsystemtype using (heatingorsystemtypeid)
ORDER BY transactiondate
GROUP BY parcelid;

SELECT p.*, MAX(transactiondate)
FROM properties_2017 as p
JOIN predictions_2017 using (parcelid)
WHERE longitude IS NOT NULL and latitude IS NOT NULL and YEAR(transactiondate) = '2017'
GROUP BY parcelid, p.id, p.airconditioningtypeid, p.architecturalstyletypeid, p.basementsqft, p.bathroomcnt, p.bedroomcnt, p.buildingclasstypeid, p.buildingqualitytypeid, p.calculatedbathnbr, p.decktypeid, p.finishedfloor1squarefeet, p.calculatedfinishedsquarefeet, p.finishedsquarefeet12, p.finishedsquarefeet13, p.finishedsquarefeet15, p.finishedsquarefeet50, p.finishedsquarefeet6, p.fips, p.fireplacecnt, p.fullbathcnt, p.garagecarcnt, p.garagetotalsqft, p.hashottuborspa, p.heatingorsystemtypeid, p.latitude, p.longitude, p.lotsizesquarefeet, p.poolcnt, p.poolsizesum, p.pooltypeid10, p.pooltypeid2, p.pooltypeid7, p.propertycountylandusecode, p.propertylandusetypeid, p.propertyzoningdesc, p.rawcensustractandblock, p.regionidcity, p.regionidcounty, p.regionidneighborhood, p.regionidzip, p.roomcnt, p.storytypeid, p.threequarterbathnbr, p.typeconstructiontypeid, p.unitcnt, p.yardbuildingsqft17, p.yardbuildingsqft26, p.yearbuilt, p.numberofstories, p.fireplaceflag, p.structuretaxvaluedollarcnt, p.taxvaluedollarcnt, p.assessmentyear, p.landtaxvaluedollarcnt, p.taxamount, p.taxdelinquencyflag, p.taxdelinquencyyear, p.censustractandblock;

SELECT p_17.parcelid, logerror, transactiondate, p.*
    FROM predictions_2017 p_17
    JOIN 
        (SELECT
        parcelid, Max(transactiondate) as tdate
        FROM predictions_2017
        GROUP BY parcelid )as sq1
    on (sq1.parcelid=p_17.parcelid and sq1.tdate = p_17.transactiondate )
    JOIN properties_2017 p on p_17.parcelid=p.parcelid
    WHERE (p.latitude IS NOT NULL and p.longitude IS NOT NULL);